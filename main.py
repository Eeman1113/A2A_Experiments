import os
import json
import time
import requests
from google import genai

class AIAgentSystem:
    def __init__(self, gemini_api_key=None, groq_api_key=None):
        # Initialize API keys
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        # Set up Gemini
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.gemini_model = "gemini-2.0-flash"
        
        # Set up Groq API endpoint (instead of using the client library)
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_model = "deepseek-r1-distill-llama-70b"
        
        # Conversation history
        self.conversation = []
        
    def gemini_agent(self, prompt):
        """Agent 1: Gemini model"""
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model, 
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Gemini Agent Error: {str(e)}"
    
    def groq_agent(self, prompt):
        """Agent 2: Groq model using direct API calls instead of the client library"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 4096,
                "top_p": 0.95
            }
            
            response = requests.post(self.groq_api_url, headers=headers, json=payload)
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                return f"Groq Agent Error: Unexpected response format - {response_data}"
                
        except Exception as e:
            return f"Groq Agent Error: {str(e)}"
    
    def determine_conversation_length(self, task):
        """Determine how many turns the agents should converse based on task complexity"""
        prompt = f"""
        You are tasked with determining the optimal number of conversation turns needed for two AI agents 
        to solve this task: "{task}"
        
        Consider:
        1. Task complexity (simple tasks need fewer turns)
        2. Need for brainstorming (creative tasks may need more turns)
        3. Need for refinement (tasks requiring precision may need more turns)
        
        Return only a number between 3 and 10 representing the optimal number of turns.
        """
        
        response = self.gemini_agent(prompt)
        try:
            turns = int(response.strip())
            return max(3, min(10, turns))  # Ensure between 3-10 turns
        except:
            return 5  # Default to 5 turns if parsing fails
    
    def generate_initial_prompts(self, task):
        """Generate initial specialized prompts for each agent"""
        gemini_prompt = f"""
        You are Agent 1 (Gemini 2.5 Pro), working collaboratively with Agent 2 (Deepseek Llama 70B) on this task:
        "{task}"
        
        Your specialties are:
        - Creative thinking and ideation
        - Structured planning
        - Considering multiple perspectives
        
        Begin by introducing yourself to Agent 2 and outline your initial thoughts on how to approach this task.
        Focus on the big picture and strategic elements.
        """
        
        groq_prompt = f"""
        You are Agent 2 (Deepseek Llama 70B), working collaboratively with Agent 1 (Gemini 2.5 Pro) on this task:
        "{task}"
        
        Your specialties are:
        - Detailed analysis and implementation
        - Technical precision
        - Pragmatic refinement of ideas
        
        Wait for Agent 1's initial thoughts, then respond with your specialized perspective.
        Feel free to build on their ideas while adding your technical insights.
        """
        
        return gemini_prompt, groq_prompt
    
    def generate_follow_up_prompts(self, conversation_history, turn_number, max_turns, task):
        """Generate follow-up prompts for continuing the conversation"""
        
        # Convert conversation history to a readable format for the models
        history_text = "\n\n".join([f"{'Agent 1 (Gemini)' if i%2==0 else 'Agent 2 (Deepseek)'}: {msg}" for i, msg in enumerate(conversation_history)])
        
        gemini_prompt = f"""
        You are Agent 1 (Gemini 2.5 Pro). Review the conversation so far about the task: "{task}"
        
        Conversation history:
        {history_text}
        
        This is turn {turn_number} of {max_turns}.
        
        {'As this is the final turn, work with Agent 2 to conclude and present a final solution or recommendation.' 
          if turn_number == max_turns else 
         'Continue the collaborative discussion, building on what has been shared so far.'}
        
        Be concise but insightful. Advance the solution forward meaningfully.
        """
        
        groq_prompt = f"""
        You are Agent 2 (Deepseek Llama 70B). Review the conversation so far about the task: "{task}"
        
        Conversation history:
        {history_text}
        
        This is turn {turn_number} of {max_turns}.
        
        {'As this is the final turn, work with Agent 1 to conclude and present a final solution or recommendation.' 
          if turn_number == max_turns else 
         'Continue the collaborative discussion, building on what has been shared so far.'}
        
        Be concise but insightful. Advance the solution forward meaningfully.
        """
        
        return gemini_prompt, groq_prompt
    
    def generate_summary(self, task, conversation):
        """Generate a summary of the collaboration and final output"""
        history_text = "\n\n".join([f"{'Agent 1 (Gemini)' if i%2==0 else 'Agent 2 (Deepseek)'}: {msg}" for i, msg in enumerate(conversation)])
        
        summary_prompt = f"""
        Review this conversation between two AI agents collaborating on the task: "{task}"
        
        Full conversation:
        {history_text}
        
        Provide:
        1. A concise summary of the key insights and ideas generated
        2. The final solution or approach recommended
        3. How the collaboration between agents enhanced the result
        
        Format your response as a structured final report.
        """
        
        return self.gemini_agent(summary_prompt)
    
    def collaborate(self, task):
        """Main function to run the collaborative process"""
        print(f"📋 Task: {task}\n")
        print("🔄 Determining optimal conversation length...")
        max_turns = self.determine_conversation_length(task)
        print(f"✅ Decided on {max_turns} turns of conversation\n")
        print("🤖 Beginning agent collaboration...\n")
        
        # Initial prompts
        gemini_prompt, groq_prompt = self.generate_initial_prompts(task)
        
        # First agent starts
        print("🔵 Agent 1 (Gemini 2.5 Pro) thinking...")
        gemini_response = self.gemini_agent(gemini_prompt)
        self.conversation.append(gemini_response)
        print(f"🔵 Agent 1 (Gemini): {gemini_response}\n")
        
        # Second agent responds
        print("🟠 Agent 2 (Deepseek Llama 70B) thinking...")
        groq_response = self.groq_agent(groq_prompt + "\n\nAgent 1 said: " + gemini_response)
        self.conversation.append(groq_response)
        print(f"🟠 Agent 2 (Deepseek): {groq_response}\n")
        
        # Continue the conversation for the determined number of turns
        for turn in range(2, max_turns + 1):
            print(f"--- Turn {turn}/{max_turns} ---\n")
            
            # Generate follow-up prompts
            gemini_prompt, groq_prompt = self.generate_follow_up_prompts(
                self.conversation, turn, max_turns, task
            )
            
            # Gemini agent's turn
            print("🔵 Agent 1 (Gemini 2.5 Pro) thinking...")
            gemini_response = self.gemini_agent(gemini_prompt)
            self.conversation.append(gemini_response)
            print(f"🔵 Agent 1 (Gemini): {gemini_response}\n")
            
            # Groq agent's turn
            print("🟠 Agent 2 (Deepseek Llama 70B) thinking...")
            groq_response = self.groq_agent(groq_prompt)
            self.conversation.append(groq_response)
            print(f"🟠 Agent 2 (Deepseek): {groq_response}\n")
        
        # Generate final summary
        print("🔄 Generating final summary and output...")
        summary = self.generate_summary(task, self.conversation)
        print("\n📊 FINAL OUTPUT:\n")
        print(summary)
        
        # Return results
        return {
            "task": task,
            "conversation_turns": max_turns,
            "conversation": self.conversation,
            "summary": summary
        }
    
    def save_results(self, results, filename=None):
        """Save the collaboration results to a file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"ai_collaboration_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Set your API keys as environment variables or pass them directly
    try:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")
        
        if not gemini_key:
            gemini_key = input("Enter your Gemini API key: ")
        if not groq_key:
            groq_key = input("Enter your Groq API key: ")
            
        system = AIAgentSystem(gemini_api_key=gemini_key, groq_api_key=groq_key)
        
        # User task
        user_task = input("Enter task for AI agents to collaborate on: ")
        
        # Run collaboration
        results = system.collaborate(user_task)
        
        # Save results
        system.save_results(results)
    except Exception as e:
        print(f"Error: {str(e)}")
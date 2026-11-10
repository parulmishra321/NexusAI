from .BaseAgent import BaseAgent
from ..EventBus import event_bus
# from groq import Groq  # Import your client

class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__("ChatAgent")

    def run(self, query: str) -> str:
        event_bus.on_status_change.emit("Thinking...")
        import ollama
        try:
            response = ollama.chat(model='llama3', messages=[
                {'role': 'system', 'content': 'You are Nexus. Be helpful and concise.'},
                {'role': 'user', 'content': query},
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)} - Is Ollama running?"

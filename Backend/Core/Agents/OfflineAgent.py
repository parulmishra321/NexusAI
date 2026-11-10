from .BaseAgent import BaseAgent
from ..EventBus import event_bus
try:
    import ollama
except ImportError:
    ollama = None

class OfflineAgent(BaseAgent):
    def __init__(self, model_name="llama3"):
        super().__init__("OfflineAgent")
        self.model = model_name

    def run(self, query: str) -> str:
        if not ollama:
            return "Ollama library not installed. Cannot run offline."
            
        event_bus.on_status_change.emit("Thinking (Offline)...")
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': query},
            ])
            return response['message']['content']
        except Exception as e:
            return f"Offline Error: {e}"

from .BaseAgent import BaseAgent
from ..EventBus import event_bus
import subprocess

class AutomationAgent(BaseAgent):
    def __init__(self):
        super().__init__("AutomationAgent")

    def run(self, query: str) -> str:
        event_bus.on_status_change.emit("Automating Task...")
        
        if "notepad" in query.lower():
            subprocess.Popen(["notepad.exe"])
            return "Opening Notepad."
        
        return "I can't handle this automation task yet."

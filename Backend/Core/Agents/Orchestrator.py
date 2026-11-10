from ..EventBus import event_bus
from .ChatAgent import ChatAgent
from .SearchAgent import SearchAgent
from .AutomationAgent import AutomationAgent
from .OfflineAgent import OfflineAgent
from .RAGAgent import RAGAgent
from .SemanticsAgent import SemanticsAgent

class Orchestrator:
    def __init__(self, offline_mode=False):
        self.offline_mode = offline_mode
        self.chat_agent = ChatAgent()
        self.search_agent = SearchAgent()
        self.automation_agent = AutomationAgent()
        self.offline_agent = OfflineAgent()
        self.rag_agent = RAGAgent()
        self.semantics_agent = SemanticsAgent()

    def process(self, query: str):
        """
        Decides which agent to use based on the query.
        """
        # Simple heuristic routing for now. 
        # In future: Use a small Classifier LLM here.
        query_lower = query.lower()

        # Prioritize offline mode global toggle if implemented behavior mandates it,
        # but here we mix agents based on intent.
        
        # Check for Computational Linguistics / Semantics triggers
        semantic_keywords = [
            "synset", "wordnet", "disambiguate", "word sense", "wsd",
            "mrs", "semantic parse", "first-order logic", "first order logic",
            " fol ", "fol formula", "fol logic", "linguistics"
        ]
        if any(kw in query_lower for kw in semantic_keywords) or query_lower.startswith("fol "):
            response = self.semantics_agent.run(query)

        # Check for RAG triggers
        elif "my notes" in query_lower or "document" in query_lower or "read file" in query_lower:
             response = self.rag_agent.run(query)

        elif "open" in query_lower or "play" in query_lower or "close" in query_lower:
             response = self.automation_agent.run(query)

        elif "search" in query_lower or "who is" in query_lower or "what is" in query_lower:
             # Fallback to online search if not generic chat
             if self.offline_mode: 
                 response = self.offline_agent.run(query)
             else:
                 response = self.search_agent.run(query)
        
        else:
             # Default Chitchat
             if self.offline_mode:
                 response = self.offline_agent.run(query)
             else:
                 response = self.chat_agent.run(query)

        # Emit the response back to GUI
        event_bus.on_message.emit("Assistant", response)
        event_bus.on_status_change.emit("Ready")

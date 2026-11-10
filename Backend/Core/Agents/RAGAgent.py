from .BaseAgent import BaseAgent
from ..EventBus import event_bus
from ..Memory.VectorDB import VectorDB
from ..Semantics.SemanticsService import SemanticsService

try:
    import ollama
except ImportError:
    ollama = None

class RAGAgent(BaseAgent):
    def __init__(self):
        super().__init__("RAGAgent")
        self.db = VectorDB()
        self.semantics = SemanticsService()

    def run(self, query: str) -> str:
        if not ollama:
            return "Ollama library not installed. Cannot perform RAG."

        event_bus.on_status_change.emit("Retrieving Context (Semantic Expansion)...")
        
        # 1. Retrieve (using semantic query expansion from WordNet)
        try:
            enriched = self.semantics.enrich_rag_query(query)
            search_query = enriched["expanded_query"] if enriched.get("expansion_terms") else query
        except Exception:
            search_query = query

        results = self.db.query(search_query, n_results=2)
        documents = results['documents'][0] if results['documents'] else []
        
        if not documents:
            return "I couldn't find any relevant notes in your documents."

        context_str = "\n\n".join(documents)
        
        # 2. Augment
        prompt = f"""
        You are an assistant. Answer the user query based strictly on the context below.
        
        Context:
        {context_str}
        
        User Query: {query}
        """
        
        event_bus.on_status_change.emit("Synthesizing Answer...")
        
        # 3. Generate
        try:
            response = ollama.chat(model="llama3", messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content']
        except Exception as e:
            return f"RAG Error: {e}"

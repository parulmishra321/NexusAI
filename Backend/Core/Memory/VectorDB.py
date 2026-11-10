import chromadb
from chromadb.utils import embedding_functions
import os

class SafeOllamaEmbeddingFunction:
    """Fallback embedding function using ollama package directly."""
    def __init__(self, model_name="llama3", url="http://localhost:11434/api/embeddings"):
        self.model_name = model_name
        self.url = url

    def __call__(self, input):
        # Accept either list of texts or single string
        texts = [input] if isinstance(input, str) else input
        import ollama
        embeddings = []
        for text in texts:
            try:
                res = ollama.embeddings(model=self.model_name, prompt=text)
                embeddings.append(res['embedding'])
            except Exception:
                # Return dummy vector if offline
                embeddings.append([0.0] * 384)
        return embeddings

class VectorDB:
    def __init__(self, collection_name="nexus_memory"):
        # Persist data to 'Data/VectorDB'
        self.db_path = os.path.join(os.getcwd(), "Data", "VectorDB")
        os.makedirs(self.db_path, exist_ok=True)

        if hasattr(chromadb, "PersistentClient"):
            self.client = chromadb.PersistentClient(path=self.db_path)
        else:
            try:
                from chromadb.config import Settings
                self.client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.db_path,
                    anonymized_telemetry=False
                ))
            except Exception:
                self.client = chromadb.Client()
        
        # Use Ollama for embeddings to avoid local torch dependency
        if hasattr(embedding_functions, "OllamaEmbeddingFunction"):
            self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
                model_name="llama3",
                url="http://localhost:11434/api/embeddings"
            )
        else:
            self.embedding_fn = SafeOllamaEmbeddingFunction(
                model_name="llama3",
                url="http://localhost:11434/api/embeddings"
            )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        """
        Add a document to the knowledge base.
        """
        if metadata is None:
            metadata = {}
        
        self.collection.upsert(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def query(self, query_text: str, n_results=3):
        """
        Retrieve relevant documents.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

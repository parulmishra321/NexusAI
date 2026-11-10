import os
from .VectorDB import VectorDB
# Optional: import PyPDF2 or similar if you want PDF support
# import PyPDF2 

class IngestionPipeline:
    def __init__(self):
        self.db = VectorDB()
        self.docs_path = os.path.join(os.getcwd(), "Data", "Documents")
        os.makedirs(self.docs_path, exist_ok=True)

    def ingest_file(self, file_path: str):
        """
        Read a file and add it to the Vector DB.
        """
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        
        content = ""
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        # elif ext == ".pdf":
        #    content = self._read_pdf(file_path)
            
        if content:
            # Chunking logic could go here (splitting long text)
            # For now, we index the whole file as one chunk (simple RAG)
            self.db.add_document(
                doc_id=filename,
                text=content,
                metadata={"source": filename}
            )
            return True
        return False
        
    def ingest_all(self):
        """Ingest all files in Data/Documents"""
        count = 0
        for filename in os.listdir(self.docs_path):
            full_path = os.path.join(self.docs_path, filename)
            if self.ingest_file(full_path):
                count += 1
        return count

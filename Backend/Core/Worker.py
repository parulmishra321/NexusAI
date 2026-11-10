from PyQt5.QtCore import QObject, pyqtSlot
from .EventBus import event_bus
from .Agents.Orchestrator import Orchestrator
from .Memory.Ingestion import IngestionPipeline

class BackendWorker(QObject):
    """
    Worker runs in a separate thread to prevent blocking the GUI.
    Listens for user queries and delegates to Orchestrator.
    """
    def __init__(self):
        super().__init__()
        self.orchestrator = Orchestrator(offline_mode=False) 
        self.ingestion_pipeline = IngestionPipeline()
        
        # Connect to the bus
        event_bus.on_user_query.connect(self.process_query)
        event_bus.on_ingest_request.connect(self.run_ingestion)

    @pyqtSlot(str)
    def process_query(self, query: str):
        """
        Slot to handle user query.
        """
        event_bus.on_status_change.emit("Thinking...")
        self.orchestrator.process(query)
        
    @pyqtSlot()
    def run_ingestion(self):
        """
        Slot to run document ingestion.
        """
        event_bus.on_status_change.emit("Ingesting documents...")
        try:
            count = self.ingestion_pipeline.ingest_all()
            msg = f"Indexed {count} documents successfully."
            event_bus.on_ingest_status.emit(msg)
            event_bus.on_message.emit("System", msg)
        except Exception as e:
            error_msg = f"Ingestion Failed: {str(e)}"
            event_bus.on_ingest_status.emit(error_msg)
            event_bus.on_message.emit("System", error_msg)
        
        event_bus.on_status_change.emit("Ready")

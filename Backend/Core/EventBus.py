from PyQt5.QtCore import QObject, pyqtSignal

class EventBus(QObject):
    """
    Central Event Bus for the application.
    Facilitates loose coupling between components using Signals and Slots.
    """
    
    # Signal emitted when a new message needs to be displayed in the GUI
    # args: (sender: str, message: str)
    on_message = pyqtSignal(str, str)
    
    # Signal emitted when the assistant's status changes (Listening, Thinking, etc.)
    # args: (status: str)
    on_status_change = pyqtSignal(str)
    
    # Signal to request microphone toggle
    # args: (active: bool)
    on_mic_toggle = pyqtSignal(bool)
    
    # Signal when a user query is finalized and ready for processing
    # args: (query: str)
    on_user_query = pyqtSignal(str)

    # Signal for specific agent actions (e.g., "Opening Browser")
    # args: (agent_name: str, action: str)
    on_agent_action = pyqtSignal(str, str)

    # Signal to trigger Ingestion
    on_ingest_request = pyqtSignal()
    
    # Signal to report Ingestion Status
    # args: (message: str)
    on_ingest_status = pyqtSignal(str)

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
        return cls._instance

# Global Singleton Instance
event_bus = EventBus()

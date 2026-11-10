import sys
import threading
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

# Import GUI
from Frontend.ModernGUI import MainWindow

# Import Backend Worker
from Backend.Core.Worker import BackendWorker

# Import EventBus for initial setup if needed
from Backend.Core.EventBus import event_bus

def main():
    print("Starting Nexus AI Enhancement...", flush=True)
    app = QApplication(sys.argv)
    
    # Create the Main Window
    window = MainWindow()
    
    # Setup Backend Worker Thread
    # We move the worker to a separate thread so it doesn't block the UI
    backend_thread = QThread()
    worker = BackendWorker()
    worker.moveToThread(backend_thread)
    
    # We need to keep a reference to worker and thread to prevent garbage collection
    window.worker = worker 
    window.worker_thread = backend_thread
    
    # Start the backend thread
    backend_thread.start()
    
    print("System Ready. Showing GUI.", flush=True)
    window.show()
    
    # Execute App
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QStackedWidget, QWidget,
    QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QLabel, QSizePolicy,
    QGraphicsDropShadowEffect, QMessageBox
)
from PyQt5.QtGui import QIcon, QMovie, QColor, QTextCharFormat, QFont, QPixmap, QTextBlockFormat, QCursor
from PyQt5.QtCore import Qt, QSize, pyqtSlot, QTimer
import sys
import os
from dotenv import dotenv_values

# Import EventBus
from Backend.Core.EventBus import event_bus

env_vars = dotenv_values(".env")
AssistantName = env_vars.get("AssistantName", "Nexus")
current_dir = os.getcwd()

# Graphics Paths
GraphicsDirPath = rf"{current_dir}\Frontend\Graphics"

def GraphicsDirectoryPath(Filename):
    return os.path.join(GraphicsDirPath, Filename)

class ChatBubble(QWidget):
    """Custom Widget for Chat Messages to make them look modern"""
    def __init__(self, text, sender="Assistant"):
        super().__init__()
        self.layout = QVBoxLayout()
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Segoe UI", 12))
        
        if sender == "User":
            self.label.setStyleSheet("""
                background-color: #0078D7; 
                color: white; 
                border-radius: 10px; 
                padding: 10px;
            """)
            self.layout.setAlignment(Qt.AlignRight)
        else:
            self.label.setStyleSheet("""
                background-color: #333333; 
                color: white; 
                border-radius: 10px; 
                padding: 10px;
            """)
            self.layout.setAlignment(Qt.AlignLeft)
            
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

class ChatSection(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Connect to Signals
        event_bus.on_message.connect(self.add_message)
        event_bus.on_status_change.connect(self.update_status)

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Top Bar Area (Sync Button)
        top_bar = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888; font-size: 10pt;")
        self.status_label.setAlignment(Qt.AlignLeft)
        
        self.sync_btn = QPushButton("Sync Docs")
        self.sync_btn.setFixedWidth(100)
        self.sync_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.sync_btn.clicked.connect(self.trigger_ingest)
        
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()
        top_bar.addWidget(self.sync_btn)
        layout.addLayout(top_bar)
        
        # Chat Display Area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            background-color: #1e1e1e; 
            color: white; 
            border: none;
            font-family: 'Segoe UI';
            font-size: 14px;
        """)
        layout.addWidget(self.chat_display)

        # Input Area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #0078D7;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084e0;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        layout.addLayout(input_layout)

    def send_message(self):
        text = self.input_field.text().strip()
        if text:
            self.append_message(f"You: {text}", "User")
            self.input_field.clear()
            event_bus.on_user_query.emit(text)

    def trigger_ingest(self):
        event_bus.on_ingest_request.emit()

    @pyqtSlot(str, str)
    def add_message(self, sender, message):
        self.append_message(f"{sender}: {message}", sender)

    def append_message(self, text, sender):
        color = "#0078D7" if sender == "User" else "#00FF7F" 
        formatted_text = f'<div style="margin: 5px;"><b style="color:{color};">{sender}:</b> <span style="color:white;">{text.split(": ", 1)[-1]}</span></div>'
        self.chat_display.append(formatted_text)
        
    @pyqtSlot(str)
    def update_status(self, status):
        self.status_label.setText(status)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nexus AI - Advanced Agentic Assistant")
        self.resize(1000, 700)
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        self.chat_screen = ChatSection()
        self.central_widget.addWidget(self.chat_screen)
        self.setStyleSheet("background-color: #121212;")

def GraphicalUserInterface():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

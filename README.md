# NexusAI – Generative AI Voice Assistant

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## Overview
NexusAI is a modular, offline-first, local AI assistant designed for privacy and low latency. It leverages an event-driven architecture to orchestrate multiple agents (Chat, Search, RAG, Automation) and uses local LLMs (via Ollama) to answer queries without relying on the cloud.

> _Demo GIF coming soon: NexusAI UI + RAG pipeline in action_

## Features
- **Multi-Agent Orchestration**: Intelligent routing between Chat, Search, and RAG agents.
- **Offline Capability**: Runs 100% locally using Llama 3 and Ollama.
- **RAG (Retrieval Augmented Generation)**: Chat with your personal documents (`.txt`).
- **Low Latency UI**: Built with PyQt5 and threaded workers for a responsive experience.
- **Modern Dark Mode**: Aesthetic GUI with real-time status updates.

## Architecture
NexusAI follows a hexagonal, event-driven architecture using a central `EventBus` to decouple the frontend from the backend logic.

![Architecture Diagram](Graphics/architecture.png)

## Agent System
- **Orchestrator**: The brain that decides which agent handles the request.
- **Search Agent**: (Placeholder) For online information retrieval.
- **Chat Agent**: Handles general conversation using local or cloud LLMs.
- **RAG Agent**: Retrieves context from local vector storage for document-based QA.
- **Automation Agent**: Executes system commands (e.g., opening apps).
- **Offline Agent**: Fallback for when no internet is available.

## Memory & VectorDB
- Uses **ChromaDB** for storing document embeddings.
- **Ollama** embeddings are used to index text files for fast semantic retrieval.

## Speech Interface
- **Text-to-Speech**: (Planned/Integrated) Support for system voice feedback.
- **Voice Activity Detection**: (Roadmap) Triggering assistant via voice.

## GUI (Frontend)
- **PyQt5**: Provides the desktop application window.
- **ModernGUI.py**: Defines the layout, styling, and signal connections.

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com/) (running `llama3`)
- Dependencies listed in `Requirements.txt`

## Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/parulmishra321/NexusAI.git
    cd NexusAI
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r Requirements.txt
    ```

4.  **Configure Environment**
    Rename `.env.example` to `.env` and add your keys (optional for offline mode).
    ```bash
    copy .env.example .env
    ```

5.  **Run the Application**
    ```bash
    python Main.py
    ```

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```ini
GROQ_API_KEY=your_key_here
OLLAMA_HOST=http://localhost:11434
OPENAI_API_KEY=your_key_here
```

## 📂 Project Structure

```
NexusAI/
├── Backend/
│   ├── Core/
│   │   ├── Agents/          # Logic for Chat, Search, RAG
│   │   ├── Memory/          # VectorDB and Ingestion
│   │   ├── EventBus.py      # Signal handling
│   │   └── Worker.py        # Background processing
├── Frontend/
│   ├── Graphics/            # Assets
│   └── ModernGUI.py         # PyQt5 Interface
├── Data/
│   ├── Documents/           # Place .txt files here
│   └── VectorDB/            # ChromaDB storage
├── Main.py                  # Entry Point
└── requirements.txt         # Dependencies
```

## 🧠 RAG Pipeline

1.  **Ingest**: `Ingest.py` reads `.txt` files from `Data/Documents`.
2.  **Embed**: Uses `Ollama` (Llama3) to create vector embeddings.
3.  **Store**: Saves embeddings locally in `Data/VectorDB` (ChromaDB).
4.  **Retrieve**: `RAGAgent` queries the DB for relevant context.
5.  **Generate**: LLM synthesizes an answer using the retrieved context.

Example Query:
"Summarize my notes about 'Financial Compliance' and highlight key risks."

## 📦 Dependencies

-   **PyQt5**: GUI Framework.
-   **Ollama**: Local LLM runner.
-   **ChromaDB**: Vector Store.
-   **Rich**: Terminal formatting.
-   **Groq**: (Optional) Cloud inference.

## 🖥️ Hardware Requirements

-   **CPU**: Modern Multi-core (Intel i5/Ryzen 5 or better).
-   **RAM**: 8GB minimum (16GB recommended for Llama3).
-   **GPU**: Optional (speeds up Ollama inference significantly).
-   **Disk**: ~10GB free space for models and environment.

## 🧪 Testing Modes

-   **Online Mode**: Uses SearchAgent (requires pending API integration).
-   **Offline Mode**: Full functionality for Chat and RAG using local models.
-   **Debug Mode**: Check terminal for `Rich` logs and `EventBus` signals.

## 📦 Packaging / Deployment

To build a standalone `.exe`:

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile Main.py
```

## Roadmap
- [ ] Voice Activity Detection
- [ ] Web Interface (FastAPI + React)
- [ ] Docker Support
- [ ] Improved Automation Agent

## Author
Created by Parul Mishra
GitHub: https://github.com/parulmishra321
LinkedIn: https://www.linkedin.com/in/parul-mishra-576a95100/

## License
MIT License — free for personal and commercial use.

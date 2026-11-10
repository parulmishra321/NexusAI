# Changelog

All notable changes to the **Nexus AI** project (formerly "Jarvis AI") will be documented in this file.

## [2.1.0] - 2026-11-10
### 🔬 Computational Linguistics & Semantic Processing Module
Added native computational semantics and formal linguistics capabilities integrated across the agent and RAG pipeline:
- **WordNet Integration (`Backend/Core/Semantics/WordNetModule.py`)**:
  - Full WordNet corpus integration via NLTK for synset retrieval, definition lookup, lexical relations, and hypernym/hyponym tracking.
  - Algorithmic Word Sense Disambiguation (WSD) implementing Simplified/Extended Lesk algorithm and Wu-Palmer semantic similarity.
- **LLM-Driven Structured Sense Selection (`Backend/Core/Semantics/LLMSenseSelector.py`)**:
  - Contextual sense disambiguation powered by local Ollama inference (`llama3`).
  - Returns strict structured JSON mapping words to exact WordNet synset IDs, definitions, confidence scores, and semantic justifications, with graceful algorithmic fallback.
- **ERG-Style MRS & First-Order Logic (`Backend/Core/Semantics/SemanticParser.py`)**:
  - Mock semantic parsing handler producing English Resource Grammar (ERG) compliant Minimal Recursion Semantics (MRS) with handles, elementary predications, and `qeq` constraints.
  - Automatic translation of MRS structures and quantifier scoping into formal First-Order Logic (FOL) formulas (Unicode and ASCII).
- **Agent & RAG Pipeline Integration**:
  - Added `SemanticsAgent` (`Backend/Core/Agents/SemanticsAgent.py`) and wired it into `Orchestrator.py` for routing natural language semantic queries.
  - Enhanced `RAGAgent.py` with semantic query expansion via `SemanticsService.enrich_rag_query()`.
- **Command-Line Interface (`semantics_cli.py`)**:
  - Rich CLI for standalone testing and interactive demonstrations (`--demo`, `--synsets`, `--wsd`, `--llm-wsd`, `--mrs`, `--fol`, `--rag-expand`).

## [2.0.0] - 2026-01-12
### 🚀 Major Transformation: "The Agentic Era"
This release marks a complete rewrite of the core architecture, transitioning from a linear script-based assistant to a modular, event-driven Multi-Agent System.

### ✨ New Features
- **Multi-Agent Orchestrator**: Introduction of a central brain (`Backend/Core/Agents/Orchestrator.py`) that intelligently routes queries to specialized agents:
    - `ChatAgent`: For general conversation (Groq/Llama3).
    - `SearchAgent`: For real-time information retrieval.
    - `AutomationAgent`: For system control (App opening/media).
    - `RAGAgent`: For talking to your personalized documents.
    - `OfflineAgent`: Setup for privacy-focused local inference using Ollama.
- **Local RAG (Retrieval Augmented Generation)**:
    - Integrated **ChromaDB** for vector storage.
    - Added **Ingestion Pipeline** (`ingest.py`) to index local `.txt` notes.
    - Fully offline embedding generation using `sentence-transformers`.
- **Modernized GUI**:
    - Completely rewritten Frontend in PyQt5 (`ModernGUI.py`).
    - Implemented Dark Mode with "Glassmorphism" aesthetics.
    - Added "Sync Docs" button for easy knowledge base updates.
- **Offline Intelligence**:
    - Added native support for **Ollama** (Llama 3) as a fallback network is unavailable.

### ⚡ Performance & Architecture
- **EventBus Architecture**: Replaced high-latency file-polling with **PyQt Signals & Slots** (`EventBus.py`). UI updates are now instant (<50ms) rather than polling every 1-2 seconds.
- **Threading**: Heavy AI tasks are now offloaded to a background `Worker` thread, ensuring the GUI never freezes.
- **Hexagonal Directory Structure**:
    - Cleaned root directory.
    - Logic isolated in `Backend/Core`.
    - Assets moved to `Frontend/Graphics`.

### 🗑️ Removals & Deprecations
- **Removed File-Based IPC**: Deleted all `.data` file dependencies (`Mic.data`, `Status.data`, `Responses.data`). The system no longer wears out your SSD reading/writing temp files for communication.
- **Deleted Legacy Scripts**: Removed monolithic `SpeechToText.py`, `TextToSpeech.py`, and `ImageGeneration.py` in favor of modular Agent classes.
- **Cleanup**: Removed `Legacy/` folder and `__pycache__` artifacts.

### 🔧 Configuration
- Added `setup.bat` for one-click dependency installation.
- Updated `Requirements.txt` with `chromadb`, `ollama`, and `rich`.

---
## [1.0.0] - Legacy
- Initial Release of "Jarvis AI".
- Monolithic `Main.py` structure.
- Basic intent matching using `cohere` classification.
- Simple synchronous execution loop.

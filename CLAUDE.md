# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot for querying course materials. It uses FastAPI backend, vanilla JavaScript frontend, ChromaDB for vector storage, and Anthropic Claude with tool calling for AI responses.

## Commands

**Always use `uv` for running the server and managing all dependencies. Never use `pip` directly.**

### Install Dependencies
```bash
uv sync
```

### Add a New Dependency
```bash
uv add <package-name>
```

### Run Development Server
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

Or use the shell script (requires Git Bash on Windows):
```bash
./run.sh
```

### Access Points
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Architecture

### Query Flow
```
Frontend (script.js)
    → POST /api/query
    → app.py
    → rag_system.py (orchestrator)
    → ai_generator.py (1st Claude call with tools)
    → search_tools.py (tool execution)
    → vector_store.py (ChromaDB semantic search)
    → ai_generator.py (2nd Claude call with results)
    → Response with sources
```

### Backend Components (`/backend/`)

| File | Purpose |
|------|---------|
| `app.py` | FastAPI endpoints: `/api/query`, `/api/courses` |
| `rag_system.py` | Main orchestrator coordinating all components |
| `ai_generator.py` | Claude API wrapper with tool execution loop |
| `search_tools.py` | `Tool` base class, `CourseSearchTool`, `ToolManager` |
| `vector_store.py` | ChromaDB with two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| `document_processor.py` | Parses course docs, extracts metadata, sentence-based chunking |
| `session_manager.py` | Conversation history per session |
| `config.py` | Configuration dataclass with env vars |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |

### Document Format (`/docs/`)

Course documents follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [lesson title]
...
```

### Key Configuration (`config.py`)
- `CHUNK_SIZE`: 800 chars per chunk
- `CHUNK_OVERLAP`: 100 chars overlap
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

## Environment Setup

Create `.env` in project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

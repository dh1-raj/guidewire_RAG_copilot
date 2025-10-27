# RAG Coding Assistant

A Streamlit-based coding assistant that uses Retrieval-Augmented Generation (RAG) to reference your documentation and generate code using OpenAI API. Embeddings are stored in Qdrant with real-time streaming responses and GitHub Copilot integration via MCP.

## Features
- **Document Upload & Embedding**: Upload `.txt`, `.md`, `.pdf`, `.docx` files; batch embeddings for performance
- **Vector Search**: Store and search document embeddings in Qdrant (Docker-based)
- **Code Generation**: Query for code generation with GPT-4
- **Real-time Streaming**: SSE-based streaming responses in Streamlit UI
- **GitHub Copilot Integration**: MCP server exposes RAG search to GitHub Copilot for in-editor code suggestions
- **Conversation History**: Maintain context across chat sessions
- **Comprehensive Logging**: Daily log files with DEBUG, INFO, ERROR levels
- **CLI Tools**: `rag` command for quick searches from terminal
- **Batch Processing**: Optimized batch embeddings (100/call) and Qdrant upserts (100 points/batch)

## Core Files

| File | Purpose |
|------|---------|
| `app.py` | Main entry point (legacy; use frontend.py) |
| `frontend.py` | Streamlit UI (upload, chat, analytics tabs) |
| `backend/main.py` | FastAPI backend (`/upload`, `/generate`, `/generate-stream` endpoints) |
| `generator.py` | GPT-4 code generation with streaming support |
| `embedder.py` | Batch embedding using OpenAI API |
| `qdrant_utils.py` | Qdrant connection, batch upsert, search utilities |
| `mcp_server.py` | MCP server for GitHub Copilot integration |
| `search_rag.py` | CLI search utility |
| `rag_cli.py` | Interactive CLI for RAG queries |
| `ask` | Quick wrapper script for `rag_cli.py` |

## Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for Qdrant)
- OpenAI API key
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Project_1
   ```

2. **Create and activate virtual environment:**
   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Qdrant (Docker):**
   ```bash
   # Option 1: Single container
   docker run -d -p 6333:6333 qdrant/qdrant:v1.12.1
   
   # Option 2: Docker Compose (recommended)
   docker-compose up -d  # if docker-compose.yml exists
   ```

5. **Configure environment variables:**
   Create `.env` in project root:
   ```env
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=
   OPENAI_API_KEY=your_openai_api_key_here
   API_URL=http://localhost:8000
   LOG_LEVEL=INFO
   ```

### Running the Application

**Terminal 1 - Start Backend API:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
streamlit run frontend.py
```

**Terminal 3 - (Optional) Start MCP Server for GitHub Copilot:**
```bash
python mcp_server.py
```

**Access UI:**
- Streamlit: http://localhost:8501
- API Docs: http://localhost:8000/docs

### CLI Usage

**Quick search (requires setup script):**
```bash
rag "event-driven architecture"
ask "kafka streams"
```

**Interactive CLI:**
```bash
python rag_cli.py
```

## Logging

Logs are saved to `rag_pipeline_YYYYMMDD.log`:

```bash
# Monitor in real-time
tail -f rag_pipeline_*.log

# Search for errors
grep "ERROR" rag_pipeline_*.log
```

## GitHub Copilot Integration

The `mcp_server.py` implements the Model Context Protocol (MCP) to expose RAG search to GitHub Copilot:

1. Start MCP server: `python mcp_server.py`
2. Configure VS Code with MCP client settings
3. Copilot suggestions will reference your knowledge base

See deployment docs for Copilot setup details.

## Project Structure

```
Project_1/
├── app.py                    # Legacy entry (use frontend.py)
├── frontend.py               # Streamlit UI
├── generator.py              # Code generation logic
├── embedder.py               # Embedding utilities
├── search_rag.py             # Search CLI
├── rag_cli.py                # Interactive CLI
├── ask                        # Quick wrapper script
├── mcp_server.py             # GitHub Copilot integration
├── backend/
│   ├── main.py               # FastAPI app
│   └── models.py             # Data models
├── requirements.txt          # Python dependencies
├── .env                       # Secrets (git-ignored)
├── .gitignore                # Git ignore rules
├── README.md                 # This file
└── rag/                       # RAG pipeline modules
    ├── document_processor.py
    └── ...
```

## Security & Git Configuration

**`.gitignore` rules (auto-generated):**
- API keys and secrets (`.env`, `.env.local`)
- Virtual environment (`venv/`, `.venv/`)
- Python cache (`__pycache__/`, `*.pyc`)
- IDE files (`.vscode/`, `.idea/`)
- Documentation (`.md`, `.txt`, except README)
- Logs (`*.log`)
- JSON config files (except safe public configs)

**Best practices:**
- Never commit `.env` or API keys
- Use `.env.example` for template (create manually)
- Store secrets in environment variables or secure vaults in production
- Review `.gitignore` before pushing

## Dependencies

Key packages (see `requirements.txt` for full list):
- `streamlit` – UI framework
- `fastapi` + `uvicorn` – Backend API
- `openai` – GPT-4 & embeddings (legacy: openai==0.28.0)
- `qdrant-client` – Vector DB client
- `sseclient-py` – SSE streaming client
- `mcp` – Model Context Protocol
- `python-dotenv` – Environment variables
- `PyPDF2`, `python-docx` – Document parsing

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Qdrant connection fails | Check Docker is running: `docker ps` |
| OpenAI API errors | Verify `OPENAI_API_KEY` in `.env` |
| Streamlit port 8501 in use | `streamlit run frontend.py --logger.level=debug --server.port 8502` |
| Slow uploads | Increase timeout in `qdrant_utils.py` or check network |

## References
- [GitHub Copilot Docs](https://docs.github.com/en/copilot)
- [OpenAI API](https://platform.openai.com/docs)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [MCP Spec](https://modelcontextprotocol.io/)

## License
MIT (or specify your license)

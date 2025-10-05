# Telecom Customer Support Bot (Memory)

**Stack**
- FastAPI backend
- Streamlit UI
- ChromaDB for KB + memory
- all-MiniLM-L6-v2 embeddings
- Ollama (default) or OpenAI

## Quickstart (Windows PowerShell)

```ps1
python -m venv venv
.env\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Ollama (free)
# install from https://ollama.com/download
# then: ollama pull mistral

# Prepare knowledge base
python ingest.py

# Run backend
python -m uvicorn backend.main:app --reload

# In a second terminal
python -m streamlit run frontend/streamlit_app.py
```
Set API URL in the sidebar to http://127.0.0.1:8000

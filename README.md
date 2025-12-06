Intelligent Document Question-Answering Chatbot (LLM + RAG)

Requirements
- Python 3.11 recommended on Windows
- A Groq API key (free): https://console.groq.com/keys

Setup
1) Clone or create the folder and open it in VS Code.
2) Create venv:
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
3) Install deps:
   pip install -r requirements.txt
   (If torch/faiss fail)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install --prefer-binary faiss-cpu
4) Add your key to .env:
   GROQ_API_KEY=gsk_...
5) Put documents into data/raw/
   data/raw/AllCombined
   data/raw/1of2/...
   data/raw/2of2/...
6) Build the vector index (one-time):
   python -m src.ingest
7) Run CLI:
   python -m src.cli --model llama3-8b-8192
   Type 'exit' to quit.
8) Or run Streamlit app:
   streamlit run src/app.py

Config knobs (src/config.py or .env)
- EMBED_MODEL: sentence-transformers/all-MiniLM-L6-v2 (default)
- CHUNK_SIZE / CHUNK_OVERLAP: control chunk granularity
- TOP_K: retrieved chunks per question
- MAX_CONTEXT_CHARS: prompt context cap

Troubleshooting
- Module import errors: ensure src/__init__.py exists and run commands as "python -m src.ingest".
- Slow ingestion: it’s one-time. Reduce CHUNK_SIZE or use a smaller embed model.
- Key errors: verify GROQ_API_KEY in .env and restart terminal/VS Code.

Notes
- Everything except the LLM runs locally and free.
- To avoid committing secrets, .env is ignored by .gitignore.
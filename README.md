# Personal Knowledge Agent

RAG app that answers questions from your own PDFs (notes, resumes, study material).

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key at https://console.groq.com and either:
- paste it into the sidebar when the app runs, or
- create a `.env` file (copy `.env.example`) and set `GROQ_API_KEY`.

## Run

```bash
streamlit run app.py
```

## How it works

1. Upload a PDF → text is extracted with `pypdf`.
2. Text is split into ~800-character overlapping chunks.
3. Chunks are embedded locally (`all-MiniLM-L6-v2`, free, no API call) and stored in a persistent ChromaDB collection (`chroma_db/` folder).
4. When you ask a question, the top 4 most relevant chunks are retrieved.
5. Those chunks + your question are sent to Groq's `llama-3.1-8b-instant` model, which answers using only that context.

## Notes

- First run downloads the embedding model (~80MB) — one-time.
- Works with text-based PDFs. Scanned/image-only PDFs need OCR first (not included).
- The vector DB persists across runs — uploaded PDFs stay indexed until you delete the `chroma_db/` folder.
"""
rag_engine.py
Core logic for the Personal Knowledge Agent:
PDF -> text -> chunks -> embeddings -> ChromaDB -> retrieval -> Groq LLM answer
"""

import chromadb
from chromadb.utils import embedding_functions
import pypdf
from groq import Groq

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "knowledge_base"

# Free, local embedding model (no API key needed for this part)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection():
    """Get (or create) the persistent vector collection."""
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def extract_text_from_pdf(file_path: str) -> str:
    """Pull all text out of a PDF file."""
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks so context isn't cut mid-idea."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def add_document(file_path: str, doc_name: str) -> int:
    """Extract, chunk, embed, and store a PDF. Returns number of chunks stored."""
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    if not chunks:
        return 0

    collection = get_collection()
    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_name, "chunk": i} for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)  





def retrieve_context(query: str, n_results: int = 4) -> list[str]:
    """Find the most relevant chunks for a question."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []


def ask_groq(query: str, context_chunks: list[str], api_key: str) -> str:
    """Send the question + retrieved context to Groq's LLM and get an answer."""
    client = Groq(api_key=api_key)
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find this in the document."
Keep the explanation simple and clear.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
   model="openai/gpt-oss-20b",

    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)

    return response.choices[0].message.content

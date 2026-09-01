"""
app.py
Streamlit UI for the Personal Knowledge Agent.
Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from rag_engine import add_document, retrieve_context, ask_groq

st.set_page_config(page_title="Personal Knowledge Agent", page_icon="🧠")
st.title("🧠 Personal Knowledge Agent")
st.caption("Upload a PDF and ask questions about it.")

# --- API key ---
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value="os.getenv("GROQ_API_KEY")",

    help="Free key from console.groq.com",
)
st.sidebar.markdown("Get a free key at [console.groq.com](https://console.groq.com)")

# --- Upload + index ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and st.button("Process Document"):
    with st.spinner("Reading and indexing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        num_chunks = add_document(tmp_path, uploaded_file.name)
        os.unlink(tmp_path)

    if num_chunks:
        st.success(f"Indexed {num_chunks} chunks from '{uploaded_file.name}'")
    else:
        st.error("Couldn't extract text — is this a scanned/image-only PDF?")

st.divider()

# --- Ask questions ---
query = st.text_input("Ask a question about your document(s)")

if query:
    if not api_key:
        st.error("Enter your Groq API key in the sidebar first.")
    else:
        with st.spinner("Searching and generating answer..."):
            context = retrieve_context(query)

        if not context:
            st.warning("No documents indexed yet. Upload and process a PDF first.")
        else:
            answer = ask_groq(query, context, api_key)
            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Source chunks used"):
                for i, c in enumerate(context):
                    st.markdown(f"**Chunk {i + 1}:** {c[:300]}...")

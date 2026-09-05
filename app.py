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

# --- API key in Sidebar ---
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.getenv("GROQ_API_KEY", ""),
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

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input bar at the bottom
query = st.chat_input("Ask a question about your document(s)")

if query:
    # 1. Check if API key is provided
    if not api_key:
        st.error("Enter your Groq API key in the sidebar first.")
    else:
        # 2. Display user query
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # 3. Generate answer and display
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                context = retrieve_context(query)

                if not context:
                    answer = "No documents indexed yet. Upload and process a PDF first."
                else:
                    answer = ask_groq(query, context, api_key)

            st.write(answer)

            if context:
                with st.expander("Source chunks used"):
                    for i, c in enumerate(context):
                        st.markdown(f"**Chunk {i + 1}:** {c[:300]}...")

        # 4. Save assistant answer to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

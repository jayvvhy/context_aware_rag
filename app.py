import streamlit as st
from dotenv import load_dotenv

from document_indexer import DocumentIndexer
from rag_engine import ContextAwareRAG

# Load .env locally (for OPENAI_API_KEY); on cloud we'll use secrets/env vars
load_dotenv()


@st.cache_resource
def load_indexer():
    indexer = DocumentIndexer(base_dir="artefacts")
    indexer.load_vector_store_from_disk()
    return indexer


def get_rag():
    """Get a per-session RAG object with independent history."""
    if "rag" not in st.session_state:
        indexer = load_indexer()
        st.session_state["rag"] = ContextAwareRAG(indexer=indexer)
        st.session_state["messages"] = []  # for UI chat history
    return st.session_state["rag"]


st.set_page_config(page_title="Stanford Admin Guide RAG", page_icon="📘")
st.title("📘 Context Aware RAG for Stanford Admin Guide")

rag = get_rag()

# ---- Clear history button ----
if st.button("🧽 Clear History"):
    rag.history.clear()
    st.session_state["messages"] = []
    st.success("Chat history cleared.")


# ---- Display previous messages ----
for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ---- Input box for new user query ----
prompt = st.chat_input("Ask a question about the Stanford Admin Guide")

if prompt:
    # Add user message to UI history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get RAG answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.query(prompt, k=3, return_context=True)
            answer = result["answer"]
            st.markdown(answer)

    # Add assistant response to UI history
    st.session_state["messages"].append({"role": "assistant", "content": answer})

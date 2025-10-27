import streamlit as st
from embedder import get_embedding
from qdrant_utils import get_qdrant_client, create_collection, upload_documents, search_documents
from generator import generate_code

st.title("RAG Coding Assistant")

QDRANT_URL = st.text_input("Qdrant URL", "https://your-qdrant-url")
QDRANT_API_KEY = st.text_input("Qdrant API Key", type="password")
OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
COLLECTION_NAME = "reference_docs"

st.header("Upload Reference Documents")
uploaded_files = st.file_uploader("Upload .txt or .md files", accept_multiple_files=True)
if uploaded_files and QDRANT_URL and QDRANT_API_KEY and OPENAI_API_KEY:
    docs = [f.read().decode("utf-8") for f in uploaded_files]
    embeddings = [get_embedding(doc, OPENAI_API_KEY) for doc in docs]
    client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
    create_collection(client, COLLECTION_NAME, len(embeddings[0]))
    upload_documents(client, COLLECTION_NAME, docs, embeddings)
    st.success("Documents embedded and stored!")

st.header("Ask for Code")
query = st.text_area("Describe the code you want")
if st.button("Generate Code") and query and QDRANT_URL and QDRANT_API_KEY and OPENAI_API_KEY:
    client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
    query_embedding = get_embedding(query, OPENAI_API_KEY)
    top_docs = search_documents(client, COLLECTION_NAME, query_embedding)
    context = "\n\n".join(top_docs)
    code = generate_code(context, query, OPENAI_API_KEY)
    st.subheader("Generated Code")
    st.code(code)

import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


st.title("📄 Chat With Your PDF")


@st.cache_resource
def create_vector_db(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Safety check
    if not docs:
        return None

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    # Safety check
    if not chunks:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db


file = st.file_uploader("Upload PDF", type="pdf")

if file:

    st.success("PDF Uploaded ✅")

    with st.spinner("Processing PDF..."):

        # create temp file with .pdf extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(file.read())
            path = temp.name

        db = create_vector_db(path)

    if db is None:
        st.error("Could not process this PDF.")
        st.stop()

    st.success("PDF Ready! Ask questions below 👇")

    question = st.text_input("Ask a question")

    if question:

        results = db.similarity_search(question, k=3)

        st.subheader("Relevant text from PDF:")

        for r in results:
            st.write(r.page_content)
            st.write("---")
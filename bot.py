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

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db


file = st.file_uploader("Upload PDF", type="pdf")

if file:

    st.success("PDF Uploaded ✅")

    with st.spinner("Processing PDF..."):

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())

        db = create_vector_db(temp.name)

    st.success("PDF Ready! Ask questions 👇")

    question = st.text_input("Ask a question")

    if question:

        results = db.similarity_search(question, k=3)

        st.subheader("Answer from document:")

        for r in results:
            st.write(r.page_content)
            st.write("---")
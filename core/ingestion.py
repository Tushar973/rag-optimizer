import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# 1. Cache the Embedding Model (Heavy load)
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Function to Process PDF with Dynamic Chunking
def process_document(uploaded_file, chunk_size, chunk_overlap):
    os.makedirs("data", exist_ok=True)
    
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Dynamic Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    
    return splits, file_path

# 3. Create Vector Store (Using FAISS for speed in RAM)
def create_vector_store(splits, embeddings):
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore
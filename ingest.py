import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def ingest_codebase():
    print("Extracting Text...")
    loader=TextLoader('sample_code.py')
    raw_documents=loader.load()

    print("Chunking....")

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks=text_splitter.split_documents(raw_documents)

    print(f"Created {len(chunks)} chunks.")

    print("translating vector space via gemini...")

    embeddings=GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-2-preview')

    vector_store=FAISS.from_documents(chunks,embeddings)

    vector_store.save_local('faiss_index')

    print("Success vector knowledge base saved to faiss_index")


if __name__ == "__main__":
    ingest_codebase()

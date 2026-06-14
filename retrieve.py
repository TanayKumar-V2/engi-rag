import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

def retrieve_and_answer(query: str):
    print("1. Loading the Vector Knowledge Base...")
    # We must use the exact same embedding model we used during ingestion
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    
    # Load the local FAISS database into memory
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True # Required by FAISS security policies for local files
    )

    print("2. Setting up the Retriever...")
    # We configure the database to only return the top 3 most relevant chunks (k=3).
    # Sending the whole codebase to the LLM would blow up the context window and cost money.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("3. Initializing the LLM (The Synthesizer)...")
    # We use Gemini 2.5 Flash. It is incredibly fast and cheap, perfect for dev tooling.
    # Temperature=0 forces the AI to be highly deterministic and factual, reducing hallucinations.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    print("4. Assembling the RAG Chain...")
    # This is the most critical part of an Engineering Hub: The System Prompt.
    # We give the AI strict rules of engagement.
    system_prompt = (
        "You are an expert engineering assistant. Use the provided context to answer the developer's question. "
        "If the answer is not in the context, say 'I cannot find the answer in the provided engineering docs.' "
        "Do not make up code, dependencies, or hallucinate answers.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Combine the retrieved documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the final chain that handles both retrieval and generation
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print(f"\nExecuting Query: '{query}'")
    
    # Invoke the pipeline with our question
    response = rag_chain.invoke({"input": query})
    
    print("\n--- Engineering Hub Answer ---")
    print(response["answer"])

if __name__ == "__main__":
    # Let's test if the AI understands the mock FastAPI code we ingested earlier
    test_query = "What parameters are required in the payload to trigger a new deployment, and what happens if the environment is unknown?"
    retrieve_and_answer(test_query)
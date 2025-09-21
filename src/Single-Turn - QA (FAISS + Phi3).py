import os
import warnings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils import setup_logger, track_time

warnings.filterwarnings("ignore")


@track_time
def single_turn_qa():
    """
    Implements a Baseline Retrieval-Augmented Generation (RAG) pipeline for document question-answering using FAISS and Microsoft's Phi-3 model.

    Steps:
        1. Checks for the presence of the required data file in the ../data/ directory.
        2. Loads the document using LangChain's TextLoader.
        3. Splits the document into manageable chunks using RecursiveCharacterTextSplitter.
        4. Generates embeddings for each chunk using HuggingFace's all-MiniLM-L6-v2 model.
        5. Stores the embeddings in a FAISS vector store for efficient retrieval.
        6. Connects to the local Ollama instance running the Phi-3 LLM.
        7. Sets up a prompt template with guardrails to ensure answers are based only on the provided context.
        8. Creates a retriever and RAG chain to connect the retriever, LLM, and prompt.
        9. Accepts a user question, retrieves relevant context, and generates an answer.
        10. Logs the process and handles cases where the answer is not found in the document.

    Returns:
        None
    """
    log = setup_logger("Single Turn - QA - RAG.")
    log.info("Implementing a Baseline RAG for Document Question-Answering with FAISS and Microsoft's Phi-3.")
    if "Data Science, Machine Learning and Artificial Intelligence.txt" not in os.listdir("../data/"):
        log.error("❗️❗️File not found in data directory.")
        return

    # Load file.
    ds_ml_ai_loader = TextLoader("../data/Data Science, Machine Learning and Artificial Intelligence.txt")
    ds_ml_ai_data = ds_ml_ai_loader.load()
    log.info("✅ File loaded successfully.")
    log.debug(f"File content - first 50 characters: {ds_ml_ai_data[0].page_content[:50]}")

    # Chunking
    # Create a text splitter to break the document into smaller pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=75)
    # Split the loaded documents.
    chunks = text_splitter.split_documents(ds_ml_ai_data)
    log.debug(f"✅ Document split into {len(chunks)} chunks.")

    # Embedding
    # Define the embedding model we want to use from HuggingFace.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create the FAISS vector store from the document chunks and their embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)
    log.debug("✅ Vector store created successfully.")

    # Connecting to the LLM.
    phi3_llm = Ollama(model="phi3")
    log.info("✅ Connected to Ollama's Phi3 model locally.")

    # Prompt template with guardrails.
    prompt_template = '''
        Answer the user's question based ONLY on the following context.
        If the answer is not found in the context, respond with "I cannot find the answer in the provided document."
        
        Context:
        {context}
        
        Question:
        {input}
        
        Answer:
    '''
    prompt = PromptTemplate.from_template(prompt_template)
    # Create a retriever from our vector store.
    retriever = vector_store.as_retriever()

    # Create the main RAG chain that connects the retriever, LLM and the prompt.
    document_chain = create_stuff_documents_chain(phi3_llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # QA session.
    question = input("Enter your question: ")
    log.info(f"❓ Asking question '{question}'.")
    response = retrieval_chain.invoke({"input": question})
    log.info(f"💬 Answer: {response['answer']}")

    fallback_message = "I cannot find the answer in the provided document."
    # Conditionally log the final message.
    if fallback_message not in response["answer"]:
        log.info("✅ RAG process completed. Final answer generated successfully.")
    else:
        log.warning("🟡 RAG process completed. Answer not found in the document.")


if __name__ == "__main__":
    single_turn_qa()

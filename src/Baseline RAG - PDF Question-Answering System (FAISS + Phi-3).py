import os
import warnings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM

from utils import setup_logger, track_time

log = setup_logger("Baseline RAG - PDF QA.")

warnings.filterwarnings('ignore')


@track_time
def baseline_rag_pdf_qa():
    """
    Runs a Baseline Retrieval-Augmented Generation (RAG) Question-Answering system on a PDF document using FAISS and Phi-3.

    This function performs the following steps:
    1. Checks for the presence of the specified PDF file in the data directory.
    2. Loads the PDF document and splits it into text chunks.
    3. Generates embeddings for the chunks using a HuggingFace model and stores them in a FAISS vector store.
    4. Sets up a retriever and connects to the Phi-3 LLM via Ollama.
    5. Defines a prompt template for question answering.
    6. Creates a RAG chain combining the retriever, prompt, and LLM.
    7. Prompts the user for a question, retrieves relevant context, and generates an answer.
    8. Logs the answer and whether it was found in the document.

    Returns:
        None
    """
    pdf_file_name = "United Nations Universal Declaration of Human Rights 1948.pdf"
    log.info("Baseline RAG - PDF Question-Answering System with FAISS and Phi-3")
    log.info(f"We are working with data from {pdf_file_name[:-4]}.")
    if pdf_file_name not in os.listdir("../data/"):
        log.error("❗❗File not found in path.")
        return

    # File loading.
    un_human_rights_document = PyPDFLoader(f"../data/{pdf_file_name}")
    un_human_rights_data = un_human_rights_document.load()
    log.info("✅ Document loaded successfully.")
    log.debug(f"First 50 characters in document: {un_human_rights_data[0].page_content[:50]}")

    # Chunking.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=75)
    chunks = text_splitter.split_documents(un_human_rights_data)
    log.debug(f"✅ Document split successfully into {len(chunks)} chunks.")

    # Embedding
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding)
    log.debug("✅ Vector store created successfully.")

    retriever = vector_store.as_retriever()

    # Connecting to an LLM.
    phi3_llm = OllamaLLM(model="phi3")
    log.info("✅ Connected to Ollama's Phi3 model successfully.")

    # Prompt template
    prompt_template = '''
        You are a data analyst. You can only answer the questions based on the information from the context supplied.
        If the answer is not found in the context, respond with "I cannot find the answer in the provided document."
        
        Context:
        {context}
        
        Question:
        {input}
        
        Answer:
    '''
    prompt = PromptTemplate.from_template(prompt_template)

    # Create a RAG chain to connect the LLM, prompt and the retriever.
    document_chain = create_stuff_documents_chain(phi3_llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    example_questions = ['What is stated as the foundation of freedom, justice, and peace in the world?',
                         'According to the declaration, what specific rights does a person have if they are charged with a penal offence?',
                         "What does Article 15 say about a person's right to a nationality?",
                         'What is the primary condition required for a marriage to be entered into, according to the declaration?']

    # PDF QA session.
    user_question = input(f"A few example questions are {example_questions}. \nEnter your question: ")
    log.info(f"❓ Asking question '{user_question}'.")
    response = retriever_chain.invoke({"input": user_question})
    log.info(f"Answer: {response['answer']}")

    fallback_message = "I cannot find the answer in the provided document."
    # Conditionally log the final message.
    if fallback_message not in response["answer"]:
        log.info("✅ RAG process completed. Final answer generated successfully.")
    else:
        log.warning("🟡 RAG process completed. Answer not found in the document.")


if __name__ == "__main__":
    baseline_rag_pdf_qa()

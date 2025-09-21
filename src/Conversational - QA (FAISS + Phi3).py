import os
import warnings

import httpx
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM

from utils import setup_logger, track_time

warnings.filterwarnings('ignore')


@track_time
def conversational_pdf_qa():
    log = setup_logger("Conversational - QA - RAG.")
    log.info("Building a Conversational RAG Chatbot with Memory and LangChain")
    data_directory = "../data/"
    pdf_file_name = "United Nations Universal Declaration of Human Rights 1948.pdf"
    log.info(f"We are working with data from {pdf_file_name[:-4]}.")

    if pdf_file_name not in os.listdir(data_directory):
        log.error("❗❗ Document not found in path.")
        return

    try:
        # Loading the document.
        un_human_rights_data = PyPDFLoader(data_directory + pdf_file_name).load()
        log.info("✅ Document loaded successfully.")
        log.debug(f"First 50 characters in the document: {un_human_rights_data[0].page_content[:50]}")

        # Chunking.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(un_human_rights_data)

        # Embedding.
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        retriever = vector_store.as_retriever()
        log.debug("✅ Vector store created successfully.")

        # Connecting to LLM.
        phi3_llm = OllamaLLM(model="phi3")
        log.info("✅ Connected to Ollama's Phi3 Model successfully.")

        qa_history_aware_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
                 You are a data analyst. You can only answer the questions based on the information from the context supplied.
                 If the answer is not found in the context, respond with "I cannot find the answer in the provided document."
                 
                 Context:
                 {context}
             """),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        rephrase_question_prompt = PromptTemplate.from_template('''
            Given the following conversation and follow-up conversation, rephrase the follow-up question to be a standalone question, in its original language.
            
            Chat history:
            {chat_history}
            
            Follow up question:
            {input}
                       
            Standalone question:
        ''')
        history_aware_retriever = create_history_aware_retriever(
            llm=phi3_llm,
            retriever=retriever,
            prompt=rephrase_question_prompt
        )

        # RAG chains.
        document_chain = create_stuff_documents_chain(phi3_llm, qa_history_aware_prompt)
        conversational_retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # Conversational QA.
        chat_history = []
        log.info("✅ Conversational RAG chain created. You can now start the conversation.")
        while True:
            user_question = input(f"Ask a question about the document (or type 'quit' to exit): ")
            if user_question.lower() == "quit":
                log.info("Exiting chat.")
                break

            log.info(f"❓ Asking question '{user_question}'")
            response = conversational_retrieval_chain.invoke({
                "input": user_question,
                "chat_history": chat_history
            })
            log.info(f"💬 Answer: {response['answer']}")

            chat_history.append(HumanMessage(content=user_question))
            chat_history.append(AIMessage(content=response['answer']))

        log.info("✅ Conversational RAG process completed successfully.")

    except httpx.ConnectError as e:
        if hasattr(e, "__cause__") and getattr(e.__cause__, "winerror", None) == 10061:
            log.error("Ollama's model is not running. Please start it with `ollama run model_name`.")
        else:
            print("A different connection error occurred:", e)


if __name__ == "__main__":
    conversational_pdf_qa()

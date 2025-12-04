import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

# Load environment variables
load_dotenv()

def formatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_response_from_chain(formatted_context, user_input, prompt, llm):
    rag_chain = (
        {"context": lambda query: formatted_context, "input": RunnablePassthrough()}
        | prompt
        | llm
        | RunnablePassthrough()
    )
    response = rag_chain.invoke({"context": formatted_context, "input": user_input})
    return response.content

@st.cache_resource(show_spinner=False)
def process_document(_uploaded_file):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(_uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load PDF from temporary file path
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        
        persist_directory = "./chroma_db"
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=hf_embeddings,
            persist_directory=persist_directory
        )
        
        return vectorstore
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Initialize LLM and prompt
llm = ChatGroq(model="qwen/qwen3-32b") # type: ignore
prompt_template = """
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
Please provide the most accurate response based on the provided question.
There is no need to say "based on the context."
Just think you are talking to a student.
Be polite.
<context>
{context}
</context>
Question: {input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# File Upload and Vectorization
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
if uploaded_file:
    vectorstore = process_document(uploaded_file)
    retriever = vectorstore.as_retriever()
    
    st.title("Chat With PDF")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            docs = retriever.invoke(user_input)
            if not docs:
                response_content = "I could not find relevant information in the document regarding your query. Please ask another question or rephrase it."
            else:
                formatted_context = formatDocs(docs)
                response_content = get_response_from_chain(formatted_context, user_input, prompt, llm)
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
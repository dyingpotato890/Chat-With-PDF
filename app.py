import streamlit as st
import random
from dotenv import load_dotenv
from langchain import hub
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

def responseGenerator():
    return random.choice(
        [
            "Enter your question.",
            "Ask your question.",
            "What would you like to know?",
        ]
    )

def formatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_response_from_chain(formatted_context, user_input):
    rag_chain = (
        {"context": lambda query: formatted_context, "input": RunnablePassthrough()}
        | prompt
        | llm
        | RunnablePassthrough()
    )
    response = rag_chain.invoke({"context": formatted_context, "input": user_input})
    return response.content

#@st.cache_data
def process_document(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    document = PyPDFLoader("temp.pdf")
    pages = document.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=hf_embeddings,
    )
    retriever = vectorstore.as_retriever()

    return retriever

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192")

# File Upload and Vectorization
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
if uploaded_file:
    retriever = process_document(uploaded_file)

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

    st.title("Chat With PDF")

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat Messages From History on App Rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_input := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant Response
        with st.chat_message("assistant"):
            # Combine all previous messages (both user and assistant) into a single context
            context = "\n\n".join([message["content"] for message in st.session_state.messages])
            full_context = context + "\n\n" + user_input

            # Retrieve relevant documents for the given context
            docs = retriever.get_relevant_documents(full_context)
            if not docs:
                response_content = "I could not find relevant information in the document regarding your query. Please ask another question or rephrase it."
            else:
                # Combine all relevant documents into one context
                formatted_context = formatDocs(docs)
                response_content = get_response_from_chain(formatted_context, user_input)

            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
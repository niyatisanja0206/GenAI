import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from tempfile import NamedTemporaryFile

# Load environment variables
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")

# Initialize embedding model
embedding_model = AzureOpenAIEmbeddings(
    openai_api_base=AZURE_OPENAI_API_BASE,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    openai_api_key=AZURE_OPENAI_API_KEY,
    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)

# Streamlit UI
st.set_page_config(page_title="GenAI MCQ Generator", layout="wide")
st.title("📘 GenAI-Based MCQ & Short Answer Generator")

# File upload
uploaded_file = st.file_uploader("Upload a GenAI-related PDF", type=["pdf"])

if uploaded_file:
    # Read PDF content
    pdf_reader = PdfReader(uploaded_file)
    raw_text = "\n".join([page.extract_text() for page in pdf_reader if page.extract_text()])

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([raw_text])

    # Embed and store in FAISS
    vector_store = FAISS.from_documents(docs, embedding_model)

    # Load LLM
    llm = AzureChatOpenAI(
        openai_api_base=AZURE_OPENAI_API_BASE,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        deployment_name=AZURE_OPENAI_LLM_DEPLOYMENT,
        model_name="gpt-4o",
        temperature=0.3,
    )

    # Initialize RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )

    # Prompt templates
    mcq_prompt = (
        "Based on the content provided, generate 5 multiple choice questions (MCQs) with 4 options each. "
        "Also provide the correct answer for each question."
    )
    short_qa_prompt = (
        "Based on the content, generate 5 short answer questions along with their ideal answers."
    )

    # Generate MCQs
    if st.button("Generate Questions"):
        with st.spinner("Generating MCQs..."):
            mcq_result = qa_chain.run(mcq_prompt)
        with st.spinner("Generating Short Answers..."):
            short_qa_result = qa_chain.run(short_qa_prompt)

        # Display Results
        st.subheader("📝 Multiple Choice Questions")
        st.markdown(mcq_result)

        st.subheader("🧠 Short Answer Questions")
        st.markdown(short_qa_result)

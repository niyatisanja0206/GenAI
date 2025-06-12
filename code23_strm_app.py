import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Azure OpenAI config
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Set Streamlit page
st.set_page_config(page_title="Azure GPT-4o Chatbot", page_icon="🤖")
st.title("Azure GPT-4o Chatbot")

# Initialize LLM only once
@st.cache_resource
def init_chain():
    llm = AzureChatOpenAI(
        openai_api_base=api_base,
        openai_api_version=api_version,
        openai_api_key=api_key,
        deployment_name=deployment_name,
        model_name="gpt-4o",
        temperature=0.9,
        max_tokens=300,
        top_p=0.9,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
Here is the conversation so far:
{chat_history}
User's current question: {question}
Respond helpfully and clearly:
"""
    )

    return LLMChain(llm=llm, prompt=prompt, memory=memory)

# Load or initialize chain
llmchain = init_chain()

# Streamlit session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_input = st.chat_input("Ask something...")
if user_input:
    # Add user input to memory and UI
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Run chain and get response
    response = llmchain.invoke({"question": user_input})
    reply = response["text"]

    # Show bot reply and store
    st.chat_message("assistant").markdown(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

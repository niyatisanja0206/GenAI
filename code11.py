from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
import os

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureChatOpenAI(
    openai_api_base=api_base,
    openai_api_version=api_version,
    openai_api_key=api_key,
    deployment_name=deployment_name,
    model_name="gpt-4o",
    temperature=0.9,
    max_tokens=10000,
    top_p=0.9,
)

prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Answer the following question.
    chat_history: {chat_history}
    question: {question}
    Answer:
    """)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    input_key="question",
    memory_key="chat_history",
    output_key="text",
    return_messages=True,
    max_token_limit=1000,  # Adjust based on your needs
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chat.")
        break

    result = chain.invoke({"question": user_input})
    print(f"Assistant: {result['text']}")
    print("-------------------------------------------------------------------------")
    print("Chat History:", memory.buffer)
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
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
    max_tokens=100,
    top_p=0.9,
    #Maximum number of tokens to generate in the completion
)

prompt = PromptTemplate(
    input_variables=["country"],
    template="What is your openion on country {country}?",
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

llmchain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

result = llmchain.invoke({"country": "India"})
print(result['text'])
print("-------------------------------------------------------------------------")
print(memory.buffer)
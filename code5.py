from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain_openai import AzureChatOpenAI  # UPDATED IMPORT
from dotenv import load_dotenv

load_dotenv()

# Use the updated AzureChatOpenAI from langchain_openai
llm_model = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)

# Define prompt with variable
my_prompt = PromptTemplate.from_template(
    "Tell me about {topic} in concise manner."
)

# Create the chain
chain = LLMChain(llm=llm_model, prompt=my_prompt)

response = chain.invoke({"topic": "cricket"})
print(response["text"])


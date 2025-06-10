from langchain_core.prompts import PromptTemplate
#from langchain.chat_models import AzureChatOpenAI
import os
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

#defined llm using azureopen ai 
llm_model= AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)

#defined prompt template
my_prompt = PromptTemplate.from_template(
    "Tell me about cricket in a concise manner"
)

#connect prompt with llm using LLMChain
#result = LLMChain(llm=llm_model,prompt=my_prompt)
#result=result.invoke({})

chain = my_prompt | llm_model
result = chain.invoke({})

#result = result.content
# Print the result
print(result)
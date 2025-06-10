from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
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

examples = [
    {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"input": "What is the capital of Germany?", "output": "The capital of Germany is Berlin."},
    {"input": "What is the capital of Italy?", "output": "The capital of Italy is Rome."},
]

examples_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Input: {question}\nOutput: {answer}",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    examples_prompt=examples_prompt,
    prefix="You are a helpful assistant. Answer the following question.",
    suffix="\nAnswer:",
    input_variables=["question"],
)

formatted_prompt = few_shot_prompt.format(question="What is the capital of Spain?")

print("Formatted Prompt:")
print(formatted_prompt)

llmchain = LLMChain(
    llm=llm,
    prompt = few_shot_prompt,
)

result = llmchain.invoke({"question":"What is the capital of Spain?"})
print(formatted_prompt)
print(result['test'])

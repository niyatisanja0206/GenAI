from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Few-shot examples
examples = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."},
    {"question": "What is the capital of Italy?", "answer": "The capital of Italy is Rome."},
]

# Template for each example
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Input: {question}\nOutput: {answer}",
)

# FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a helpful assistant. Answer the following question.",
    suffix="Input: {question}\nOutput:",
    input_variables=["question"],
)

# Updated LLM (correct model_kwargs for non-standard params)
llm = AzureChatOpenAI(
    api_key=api_key,
    deployment_name=deployment_name,
    model="gpt-4o",  
    temperature=0.9,
    max_tokens=100,
    openai_api_version=api_version,
    openai_api_base=api_base,  
    model_kwargs={"top_p": 0.9},  
)

# ✅ New LangChain v0.2+ execution: prompt | llm
chain = few_shot_prompt | llm

# Run the chain
question = "What is the capital of Spain?"
result = chain.invoke({"question": question})

# Print the output
print("Result:\n", result.content)

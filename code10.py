from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables
load_dotenv()

# Load Azure OpenAI API config
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize the LLM
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

# Memory to store conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# General-purpose prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""

Here is the conversation so far:
{chat_history}
User's current question: {question}
Respond helpfully and clearly:
"""
)

# Create the chain
llmchain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

# Interactive chatbot loop
print("Chatbot initialized. Type 'exit' to quit.\n")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    response = llmchain.invoke({"question": user_input+str(memory.buffer)})
    print(f"Bot: {response['text']}\n")
    print("")
    print("--------------------------------------------------------------------------")
    print("")
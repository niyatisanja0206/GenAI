from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import List, Dict, Any
from dotenv import load_dotenv
from dataclasses import dataclass, field
import os

# Load environment variables from .env
load_dotenv()

# Setup Azure OpenAI API credentials
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
    model_name="gpt-4o",  # Or "gpt-35-turbo"
    temperature=0.9,
    max_tokens=10000,
    top_p=0.9,
)

# Define a prompt template
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer the following question.
chat_history: {chat_history}
question: {question}
Answer:
""")

# ✅ Custom memory class that avoids Pydantic restrictions
@dataclass
class SimpleCustomMemory(BaseMemory):
    chat_history: List[BaseMessage] = field(default_factory=list)

    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        history_str = "\n".join(
            f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
            for m in self.chat_history
        )
        return {"chat_history": history_str}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        self.chat_history.append(HumanMessage(content=inputs["question"]))
        self.chat_history.append(AIMessage(content=outputs["text"]))

    def clear(self) -> None:
        self.chat_history.clear()

# Initialize custom memory
memory = SimpleCustomMemory()

# Create the LLM chain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chat.")
        break

    result = chain.invoke({"question": user_input})
    print(f"Assistant: {result['text']}")
    print("-------------------------------------------------------------------------")
    print("Chat History:")
    print(memory.load_memory_variables({})["chat_history"])

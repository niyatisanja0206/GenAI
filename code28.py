from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------------------
# Define arithmetic tools
# ----------------------------
@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

@tool
def subtract(x: int, y: int) -> int:
    """Subtract two numbers."""
    return x - y

@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

@tool
def divide(x: int, y: int) -> float:
    """Divide two numbers."""
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

@tool
def power(x: int, y: int) -> int:
    """Raise x to the power of y."""
    return x ** y

@tool
def modulo(x: int, y: int) -> int:
    """Return the remainder of x divided by y."""
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x % y

# ----------------------------
# Initialize LLM
# ----------------------------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    model="gpt-4o",
    temperature=0.7,
)

# ----------------------------
# Define tools and prompt
# ----------------------------
tools = [add, subtract, multiply, divide, power, modulo]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can perform arithmetic operations."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ----------------------------
# Create agent and executor
# ----------------------------
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------
# Run the agent with input
# ----------------------------
result = agent_executor.invoke({
    "input": "What is the result of adding 5 and 3, subtracting 2, multiplying by 4, "
             "dividing by 2, raising to the power of 3, and then divide by 5 and give the reminder?"
})

# Print final result
print("\nFinal Result:\n", result)

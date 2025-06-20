import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import YahooFinanceNewsTool
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from typing import Annotated, TypedDict
import operator
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
 
load_dotenv()

# Fetch environment variables
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")

# Initialize AzureChatOpenAI model
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    openai_api_version=api_version,
    openai_api_key=api_key,
    openai_api_base=api_base,
    model_name="gpt-4o",
    temperature=0.7,
)

search = DuckDuckGoSearchRun()
finance = YahooFinanceNewsTool()

tools = [search, finance]
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content=
"""
You are a highly skilled stock analyst. Your only task is to analyze a company’s historical performance, current news about the company, recent stock data, and the latest financial/news developments. 
Based on this analysis, also tell news with the latest date and year, and then
answer the question: “Should I buy this stock?”
Respond with:
Yes or No
A concise, evidence-based conclusion (e.g., strong earnings, growth potential, or high risk, weak fundamentals, etc.)
Do not respond to any other topic or task. Focus only on stock analysis and investment recommendation.
"""
)

# Reasoning function
def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Define the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")

# Compile graph
react_graph = builder.compile()

# Compile graph
react_graph = builder.compile()

# Input message
messages = [HumanMessage(content="Should I buy the stocks of Reliance if I am in India?")]

# Invoke graph
result = react_graph.invoke({"messages": messages})

# Print results
for m in result['messages']:
    m.pretty_print()
from langchain_core.tools import tool

@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

result = add.invoke({"x": 2345, "y": 6789})
print(result)
print(add.name)
print(add.description)
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Tell me about {topic} in a concise manner"
)

result = prompt.format(topic="Python programming Language")
print(result)
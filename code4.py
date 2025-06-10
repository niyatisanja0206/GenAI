from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Tell me about {topic} in a concise manner and in {emotion}"
)

result = prompt.format(topic="Python programming Language", emotion="soft tone")
print(result)
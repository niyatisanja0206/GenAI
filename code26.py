from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wikipedia.run("Dilwale Dulhania Le Jayenge")

print(result)
print(wikipedia.name)
print(wikipedia.description)
print(wikipedia.args_schema)

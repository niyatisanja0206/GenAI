from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

result = search_tool.run("Air India Crash in Ahemdabad")
print(result)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args_schema)
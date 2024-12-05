from langchain_core.tools import tool
from langchain_ollama import ChatOllama

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

llm = ChatOllama(model="llama3.2", temperature=0)

tools = [add, multiply]
llm_forced_to_multiply = llm.bind_tools(tools, tool_choice="multiply")
out = llm_forced_to_multiply.invoke("what is 2 + 4")
print(out)
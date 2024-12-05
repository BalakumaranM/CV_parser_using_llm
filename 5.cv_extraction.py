from unstructured.partition.pdf import partition_pdf 
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel,Field
import json
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from loguru import logger
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import json
from pydantic import BaseModel, Field
from typing import Optional
from langgraph.prebuilt import ToolNode, tools_condition
import fitz
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

class CVInput(BaseModel):
    filename: str = Field(description="filename of the CV")

def extract_text_with_positions(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text+= page.get_text()
    pdf_document.close()
    return text

@tool("cv_classification", args_schema=CVInput, return_direct=True)
def cv_classification(filename: str) -> dict:
    """Useful when need to extract the CV from pdf file to a JSON Format"""
    print(f"{'*'*15} Tool call {'*'*15}")
    cv = extract_text_with_positions(filename)
    # Generate and print the description template
    # template = get_description_template(KeyValueExtraction)
    # format = json.dumps(template, indent=4)
    extract_template = """
        Given the CV information {cv} of a person. I want you to create:
        Output response must always be in the following JSON format. Return '' (empty string) for the respective key, if value not present in cv.
        
        {{
            "name": "Name of the person in cv. Return None if not present",
            "email": "Email ID of the person in cv. Return None if not present",
            "skills": "Skillset of the person in cv. Return None if not present",
            "experience": [
                {{
                "company": "Name of the company. Return None if not present",
                "years_of_experience": "Duration of experience in years. Return None if not present",
                "works_done": "Summary of work responsibilities or achievements. Return None if not present"
                }},
                ...
            ],
            "education": [
                {{
                "college": "Name of the institution. Return None if not present",
                "degree": "Degree obtained. Return None if not present",
                "cgpa": "Cumulative Grade Point Average or equivalent. Return None if not present",
                "years": "Duration of study or graduation year. Return None if not present"
                }},
                ...
            ],
            "mobile_number": "Mobile Number of the person given in the cv. Return None if not present"
            }}
    """
    output_parser = JsonOutputParser()
    extract_prompt_template = PromptTemplate(input_variables=['cv'], template=extract_template, output_parser=output_parser)# , partial_variables=format
    chain = extract_prompt_template | llm
    res = chain.invoke(input={'cv':cv})
    content = res.content
    start_index, end_index = content.index('{'), content.rindex('}')+1
    content = content[start_index:end_index]
    try:
        content_dict = json.loads(content)
        return content_dict
    except Exception as e:
        return TypeError(f"The extracted data {res.content} is not parsed in the given format")
    

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    print('\n\n', llm_with_tools.invoke(state["messages"]), '\n\n')
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


if __name__ == "__main__":

    print('Hello LangChain!') 
    filename = "CV.pdf"
    mytool = cv_classification
    # tool_node = BasicToolNode(tools=[mytool])
    tool_node = ToolNode(tools=[mytool])
    graph_builder = StateGraph(State)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile()



    # display(
    #     Image(
    #         graph.get_graph().draw_mermaid_png(
    #             curve_style=CurveStyle.LINEAR,
    #             node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
    #             wrap_label_n_words=9,
    #             output_file_path="./graph.png",
    #             draw_method=MermaidDrawMethod.PYPPETEER,
    #             background_color="white",
    #             padding=10,
    #         )
    #     )
    # )
    llm = ChatOllama(model="llama3.2", temperature=0)
    # Modification: tell the LLM which tools it can call
    llm_with_tools = llm.bind_tools([mytool])
    message = f"""You are an expert in document extraction, You have to extract the Document in the Given Format perfectly.
                Perform CV classification on the document filename: {filename}.
                Call the tool function cv_classification to extract the JSON data from the {filename},
                Recall the tool_function until get the data in desired json format. Don't halucinate.
                The format should be the below given JSON. Your response should only contain the formatted JSON output
                {{
                    'name': 'Name of the person in cv. Return None if not present',
                    'email': 'Email ID of the person in cv. Return None if not present',
                    'skills': 'Skillset of the person in cv. Return None if not present',
                    'experience': [
                        {{
                        'company': 'Name of the company. Return None if not present',
                        'years_of_experience': 'Duration of experience in years. Return None if not present',
                        'works_done': 'Summary of work responsibilities or achievements. Return None if not present'
                        }},
                        ...
                    ],
                    'education': [
                        {{
                        'college': 'Name of the institution. Return None if not present',
                        'degree': 'Degree obtained. Return None if not present',
                        'cgpa': 'Cumulative Grade Point Average or equivalent. Return None if not present',
                        'years': 'Duration of study or graduation year. Return None if not present'
                        }},
                        ...
                    ],
                    'mobile_number': 'Mobile Number of the person given in the cv. Return None if not present'
            }}
            """
    output = graph.invoke({'messages':[HumanMessage(content=message)]})
    print("\nFinal Output:")
    content = output['messages'][-1].content
    print(content, type(content))
    start_index, end_index = content.index('{'), content.rindex('}')+1
    content = content[start_index:end_index]
    content = json.loads(content)
    print(content, type(content))


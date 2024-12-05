from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

#Just using PyPDFLoader, which is very bad pdf loader and giving wrong results for Extracted Data

if __name__ == "__main__":
    # load_dotenv()
    print('Hello LangChain!')
    

    extract_template = """
        given the CV information {information} of a person. I want you to create:
        1. Extract the name of the person
        2. Extract the e-mail of the person
    """
    
    extract_prompt_template = PromptTemplate(input_variables=['information'], template=extract_template)
    llm = ChatOllama(model="llama3.2")
    chain = extract_prompt_template | llm | StrOutputParser()
    loader = PyPDFLoader("./CV.pdf")
    pages = loader.load_and_split()
    print(pages, '\n\n')
    res = chain.invoke(input={'information':pages})
    print(res)


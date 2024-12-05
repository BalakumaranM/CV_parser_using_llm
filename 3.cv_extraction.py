from unstructured.partition.pdf import partition_pdf 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

#Improvements:
#It uses the unstructured app , which can be installed using pip install, it requires tesseract (which is being used
# in the backend for extraction) and also requires poppler for pdftoimage conversion. It is completely free
# version (Apache 2.0 License)

#Disadvantages:
#It is not extracting information as key, value pairs (dictionary), to use them for further processes


def load_cv(filename):
    elements = partition_pdf(filename=filename)
    cv = "\n\n".join([str(el) for el in elements])
    return cv

if __name__ == "__main__":
    # load_dotenv()
    print('Hello LangChain!')
    cv = load_cv("CV.pdf")
    extract_template = """
        given the CV information {cv} of a person. I want you to create:
        1. Extract the name of the person
        2. Extract the e-mail of the person
    """
    
    extract_prompt_template = PromptTemplate(input_variables=['cv'], template=extract_template)
    llm = ChatOllama(model="llama3.2")
    chain = extract_prompt_template | llm | StrOutputParser()
    res = chain.invoke(input={'cv':cv})
    print(res)


from unstructured.partition.pdf import partition_pdf 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


def cv_classification(filename):
    elements = partition_pdf(filename=filename)
    cv = "\n\n".join([str(el) for el in elements])
    extract_template = """
        given the CV information {cv} of a person. I want you to create:
        Output response must always be in the following JSON format. return '' (empty string) for the respective key, if value not present in cv.
        {{
            "name": "Extract the name of the person",
            "e-mail": "Extract the e-mail of the person",
            "skillset": "Extract skillset of the person in cv",
            "experience": "Extract experience of the person given in the cv",
            "education": "Extract education of the person given in the cv",
            "mobile_number": "Extract mobile number of the person in the cv"
        }}
    """
    
    extract_prompt_template = PromptTemplate(input_variables=['cv'], template=extract_template)
    llm = ChatOllama(model="llama3.2")
    chain = extract_prompt_template | llm | StrOutputParser()
    res = chain.invoke(input={'cv':cv})
    return res



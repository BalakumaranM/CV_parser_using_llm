from unstructured.partition.pdf import partition_pdf 
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers.structured import (StructuredOutputParser, ResponseSchema)
from langchain_core.output_parsers import JsonOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
import json
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from loguru import logger
from langchain.output_parsers import RetryOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
#Improvements:
#It is extracting the information in structured manner using output parser in langchain

#Disadvantages:
#It is not operating in agentic manner to check if the output is good.


# LangChain provides the FileCallbackHandler to write logs to a file. The FileCallbackHandler is similar to the StdOutCallbackHandler, but instead of printing logs to standard output it writes logs to a file.
# We see how to use the FileCallbackHandler in this example. Additionally we use the StdOutCallbackHandler to print logs to the standard output. It also uses the loguru library to log other outputs that are not captured by the handler.
logfile = "output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler_1 = FileCallbackHandler(logfile)
handler_2 = StdOutCallbackHandler()

def load_cv(filename):
    elements = partition_pdf(filename=filename)
    cv = "\n\n".join([str(el) for el in elements])
    return cv

class Experience(BaseModel):
    company: str = Field(description="Name of the company. Return None if not present")
    years_of_experience: str = Field(description="Duration of experience in years. Return None if not present")
    works_done: str = Field(description="Summary of work responsibilities or achievements. Return None if not present")

class Education(BaseModel):
    college: str = Field(description="Name of the institution. Return None if not present")
    degree: str = Field(description="Degree obtained. Return None if not present")
    cgpa: str = Field(description="Cumulative Grade Point Average or equivalent. Return None if not present")
    years: str = Field(description="Duration of study or graduation year. Return None if not present")


# Define your desired data structure.
class KeyValueExtraction(BaseModel):
    name: str = Field(description="Name of the person in cv. Return None if not present")
    email: str = Field(description="Email ID of the person in cv. Return None if not present")
    skills: list[str] = Field(description="Skillset of the person in cv. Return None if not present")
    experience: list[Experience] = Field(description="Experience given in the cv. Return None if not present")
    education: list[Education] = Field(description="Educational background of the person given in the cv. Return None if not present")
    mobile_number: str = Field(description="Mobile Number of the person given in the cv. Return None if not present")

if __name__ == "__main__":
    # load_dotenv()
    print('Hello LangChain!')
    cv = load_cv("Resume_Dinesh.pdf")
    llm = ChatOllama(model="llama3.2", temperature=0)
    # response_schemas = [
    #     ResponseSchema(
    #         name="Given the cv, return a dictionary of key-value pairs",
    #         description="Given the CV, Extract and map values for the given keys",
    #         type= "dict"
    #     )
    # ]
    # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = PydanticOutputParser(pydantic_object=KeyValueExtraction)
    format_instructions = output_parser.get_format_instructions()
    
    
    extract_template = """
        Given the CV information of a person. Extract the informations given in the CV, map the values with the respective keys in the below mentioned format instructions.
        Don't give your response as a code of how to do the process. Dont give back only the pydantic json without filling the data from the given CV.
        Your Response must only be the json output of the extracted data from CV in the format mentioned below.
        {format_instructions}
        CV Data : 
        {cv}
        Response:"""
    # Remember if any of the above mentioned key doesn't have value, then return null.
    # And Add more keys if such present in the cv other than the listed keys
    
    extract_prompt_template = PromptTemplate(input_variables=['cv'], template=extract_template,
                                             partial_variables={"format_instructions": format_instructions})
    prompt_value = extract_prompt_template.format_prompt(cv=cv)

    chain = extract_prompt_template | llm | output_parser
    main_chain = RunnableParallel(
    completion=chain, prompt_value=extract_prompt_template
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    bad_response = "{'college':{'description':'Name of the institution'} ...}"
    retry_parser = RetryOutputParser.from_llm(parser=output_parser, llm=llm)
    retry_parser.parse_with_prompt(bad_response, prompt_value)
    res = main_chain.invoke({'cv':cv}, {"callbacks": [handler_1, handler_2]})
    logger.info(res)
    print(json.dumps(res, indent=4))


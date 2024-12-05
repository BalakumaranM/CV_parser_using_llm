import fitz
import json
import pprint
from ollama import chat
from ollama import ChatResponse

def extract_text_with_positions(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text+= page.get_text()
    pdf_document.close()
    return text

def send_to_llm(full_html):

    master_prompt = """You are a helpful assistant to create a structured data"""
    # Prepare the prompt for the LLM
    prompt = """Given the CV information of a person as a list of html pages.
    You have to fill those text blocks in a string by providing the right space as of the html.
    You should not halucinate. Should not provide information from your knowledge,You should not summarize the details. The response should contain only
    the combined blocks of text at the given poisiton in the string format.
    Below is the data of every page data of the :\n"""
    for i in range(len(full_html)):
        prompt += f"Html Page {i+1}: {full_html[i]}"

    # Send the prompt to the LLM and process the response
    # (Replace this with your specific LLM interaction method)
    response: ChatResponse = chat(model='llama3.2', messages=[
                                {
                                    'role': 'user',
                                    'content': prompt,
                                },
                                ])
    print(response)
    # structured_data = json.loads(response)  # Assuming the LLM returns JSON

    return None


pdf_path = "CV.pdf"
text = extract_text_with_positions(pdf_path)
# structured_data = send_to_llm(text_blocks=text_blocks)
print(text)
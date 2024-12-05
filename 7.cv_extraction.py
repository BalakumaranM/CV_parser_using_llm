import fitz
from ollama import chat, ChatResponse
import json




def require_master_to_compose(responses: list[dict]):
    for response in responses:
        if type(response)!=dict:
            return False
    return True

def response_validator(content):
    print('\n\nfinal content :', content, '\n\n')
    if content:
        try:
            content_json = json.loads(content).get('response')[0]
        except:
            return {'pass':False, 'error':'Output json does not contain key "response"'}

        if not content_json:
            return {'pass':False, 'error':'Output json does not contain key "content"'}
        if content_json.get('content'):
            data = content_json.get('content')
            if not data.get('name'):
                return {'pass':False, 'error':'Output json does not contain key "name"'}
            if not data.get('email'):
                return {'pass':False, 'error':'Output json does not contain key "email"'}
            return {'pass':True}
        else:
            return {'pass':False, 'error':'Output json does not contain key "content"'}
    else:
        return {'pass':False, 'error':'Output json does not contain data'}



def cv_extractor(filename: str)-> str:
    pdf_document = fitz.open(filename)
    print('\n<<<<<<<<<<<< inside cv_extractor >>>>>>>>\n')
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text+= page.get_text()
    pdf_document.close()
    # print('\ntext : ',text,'\n' )
    extract_template = f"""
        Given the CV information of a person:
        {text}. 
        I want you to create an Output response,
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
    response: ChatResponse = chat(model='llama3.2', 
                                  messages=[{"role": "user", "content": extract_template}],
                                  format="json",
                                  options={"temprature":0}
                                  )
    print("\ntool response : ",response, '\n')
    extracted_information = response['message']["content"]
    # print('\nextracted_information : ',extracted_information,'\n');exit()
    return {"extracted data":extracted_information}


all_available_functions = {
    "cv_extractor":cv_extractor
}



system_prompt = f"""You are an expert in document extraction and parsing. You have to extract and parse the CV Document in the following given format perfectly.
                User will provide the filename of the CV document.
                Always call the tool function cv_extractor to extract the JSON data from the file.
                Recall the tool_function until get the data in desired json format. Don't halucinate.
                The format should be the below given JSON. Your response should only contain the formatted JSON output
                {{
                    'response': [
                        {{  'type': 'json',
                            'content': 
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
                                }},
                            'need_tool': ' "YES" or "NO" '
                                    Yes if need to use tool function to respond to the query.
                                    No otherwise ( if already got the parsed data from tool function).
                        }}
                        ]    
                }}
                Important Note: You must not Halucinate, you must not provide data from your memory, rather from tool function response
            """

tools = [
            {
                "type":"function",
                "function":{
                    "name":"cv_extractor",
                    "description":"A tool for performing CV extraction given a filename",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description":"Filename of the CV file"}
                        },
                    "required": ["filename"]
                    }
                }
            }

        ]


filename = "Resume_Dinesh.pdf"
initial_messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content":f"The filename of the CV is '{filename}', can you extract the CV details ?"}
            ]

messages = initial_messages
return_in_json = False

retry_count = -1
max_retry = 1
while retry_count < max_retry:
    if retry_count == -1:
        response: ChatResponse = chat(model='llama3.2', 
                                    messages=messages,
                                    tools= tools,
                                    format="json",
                                    options={"temprature":0}
                                    )
    else:
        response: ChatResponse = chat(model='llama3.2', 
                                    messages=messages,
                                    format="json",
                                    options={"temprature":0}
                                    )

    
    output_message = response['message']
    retry = False
    print("output_message : ",output_message, '\n\n')
    if output_message['content']:
        print(output_message['content'])
        responses = json.loads(output_message['content'])
        for response in responses.get("response"):
            if response.get("type").lower()=='json' and "need_tool" in response and response.get("need_tool").lower()=="yes":
                retry = True
                break
        
        if retry:
            messages.append({"role":"assistant", "content": output_message['content']})
            messages.append({"role": "user", "content": "You did not used the tool to generate the response. Please regenerate the response again by utilizing the tool."})
            retry_count +=1
    
    elif output_message['tool_calls']:
        tool_calls = output_message['tool_calls']

        while tool_calls:
            messages.append({
                                "role":"assistant",
                                "tool_calls": tool_calls

                            })
            function_responses = []
            for tool_call in tool_calls:
                try:
                    tool_function_name = tool_call['function']['name']
                    tool_function_args = tool_call['function']['arguments']
                    print('\n\nfunction name :',tool_function_name,'\t',"function args :", tool_function_args,'\n\n')
                    if tool_function_name in all_available_functions:
                        error = None
                        
                        function_response = all_available_functions[tool_function_name](**tool_function_args)
                        function_responses.append(function_response)
                    else:
                        raise Exception(f"Unknown function name '{tool_function_name.replace('.','')}' ")
                except Exception as e:
                    print('hi')
                    error = f"Getting error: {e}. Please recall the function with proper modifications."
                
                messages.append({
                    "role": "tool",
                    "name": tool_function_name,
                    "content": json.dumps(function_response) if not error else error
                })
            print('\n\n function responses :', function_responses, '\n\n')
            if error or require_master_to_compose(function_responses):
                response_to_tool_call: ChatResponse = chat(model='llama3.2', 
                                  messages=messages,
                                  tools= tools,
                                  format="json"
                                  ) 

                print('\n\n composed response : ',response_to_tool_call, '\n\n')
                tool_calls = response_to_tool_call['message'].get('tool_calls', None)
                if tool_calls:
                    continue
                else:
                    final_response = response_to_tool_call
                    content = final_response['message']['content']
                    validator = response_validator(content)
                    if validator['pass']:
                        break
                    else:
                        error = validator['error']
                        print('\n\nerror :',error, '\n\n')
                        messages.append({"role":"assistant", "content": response_to_tool_call['message']['content']})
                        messages.append({
                            "role": "user",
                            "content": error
                        })
                        retry = True
    else:
        raise ValueError("No response from master LLM")
    
    if retry:
        retry_count+=1
    else:
        break

print("\n\nfinal response : ",json.loads(final_response['message']['content']))
from pydantic import BaseModel, Field
from typing import List, Any
from typing import Optional
import json

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





# Function to extract descriptions
def get_description_template(model: BaseModel) -> Any:
    template = {}
    for field_name, field in model.__fields__.items():
        if issubclass(field.type, BaseModel):  # Nested object
            template[field_name] = get_description_template(field.type)
        elif field.type == List:  # List of objects
            item_type = field.type_.__args__[0]
            if issubclass(item_type, BaseModel):
                template[field_name] = [get_description_template(item_type)]
            else:
                template[field_name] = field.field_info.description
        else:
            template[field_name] = field.field_info.description
    return template


# Generate and print the description template
template = get_description_template(KeyValueExtraction)

format = json.dumps(template, indent=4)
print(format)
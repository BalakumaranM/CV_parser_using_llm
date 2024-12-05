from typing import Annotated

# Validation function
def validate_non_empty(input_list: list):
    if not input_list:
        raise ValueError("List cannot be empty")
    return input_list

# Annotated type with a validation function as metadata
ValidatedList = Annotated[list, validate_non_empty]

# Example of using the annotated type
try:
    # A valid list
    my_valid_list: ValidatedList = validate_non_empty(["item1", "item2"])
    print(f"Validated List: {my_valid_list}")
    
    # An invalid list (empty)
    my_invalid_list: ValidatedList = [validate_non_empty([])]
except ValueError as e:
    print(f"Validation Error: {e}")

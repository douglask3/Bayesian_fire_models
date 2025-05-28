import json
import sys
sys.path.append('libs/')
sys.path.append('fire_model/')
sys.path.append('link_distribution/')

from constrain_cubes_standard import *
from FLAME import FLAME
from ConFire import ConFire
from MaxEnt import MaxEnt
from zero_inflated_logit import zero_inflated_logit
from zero_inflated_logit_global_normal import zero_inflated_logit_global_normal
from normal_ import normal_

from pdb import set_trace


def write_variables_to_namelist(variables, output_file):
    """Write a dictionary of variables to a file in the specified format.
    Inputs:
        variables (dict): A dictionary containing variable names as keys 
            and their values as values.
        output_file (str): The name of the file to which the variables will be written.
    Returns:
        None
    Example:
        my_variables = {
            "variable1": "Hello",
            "variable2": 42,
            "variable3": [1, 2, 3],
            "variable4": 3.14,
            "variable5": "World",
            "variable6": [4, 5, 6]
        }
        write_variables_to_file(my_variables, 'variables.txt')
    """

    with open(output_file, 'w') as file:
        for variable_name, variable_value in variables.items():
            # Check if the variable is a list
            if callable(variable_value):
                # If the variable is a function, save its name as a string
                file.write(f'{variable_name}:: {variable_value.__name__}\n')
            #elif isinstance(variable_value, dict):
                # If the variable is a dictionary, serialize it to JSON format
            #    variable_value = json.dumps(variable_value)
            elif isinstance(variable_value, str):
                # If the variable is a string, enclose it in quotes
                file.write(f"{variable_name}:: '{variable_value}'\n")
            elif isinstance(variable_value, list):
                # If the variable is a list, format it with brackets
                file.write(f'{variable_name}:: {variable_value}\n')
            else:
                # For other variable types, write without quotes or brackets
                file.write(f"{variable_name}:: {variable_value}\n")

def read_variables_from_namelist(file_name):
    """Read variables from a file and create them with their original names.
    Inputs:
        file_name (str): The name of the file containing the variables.
    Returns:
        dict: A dictionary of variable names and their values.
    Example Usage:
        file_name = 'variables.txt'
        read_variables = read_variables_from_file(file_name)
        # Create variables with their original names and assign the values
        for variable_name, variable_value in read_variables.items():
            exec(f"{variable_name} = {variable_value}")
        # Now you have the variables with their original names and values
        print(variable1)  # Output: Hello
        print(variable2)  # Output: 42
    """
    variables = {}
    
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.strip().split("::")
            if len(parts) == 2:
                variable_name = parts[0].strip()
                variable_value = parts[1].strip()
                if callable(variable_value):
                    # If the variable is a function, save its name
                    variable_value_set = variable_value
                elif variable_value.startswith('"') and variable_value.endswith('"'):
                    # If the variable is a string, remove the quotes
                    variable_value_set = variable_value[1:-1]
                elif variable_value.startswith('[') and variable_value.endswith(']'):
                    # If the variable is a list, parse it
                    try:        
                        variable_value_set = eval(variable_value)
                    except:
                        functions = variable_value.split(', ')
                        def define_function(fun):
                            return eval(fun.split('function ')[1].split(' at ')[0])
                        variable_value_set = [define_function(fun) for fun in functions]
                else:
                    try:
                        # Try to parse the variable as a dictionary
                        variable_value_set = eval(variable_value)
                    except (SyntaxError, NameError):
                        # If parsing fails, assume it's a non-string, non-list variable
                        variable_value_set = variable_value
                #set_trace()
                if variable_name in variables:
                    if type(variables[variable_name]) is list:
                        variables[variable_name].append(variable_value_set)
                    else:
                        variables[variable_name] = [variables[variable_name], 
                                                    variable_value_set]
                else:
                    variables[variable_name] = variable_value_set
    return variables


def read_variable_from_namelist_with_overwite(file_name, **kwargs):

    def merge_variables(dict1):
        merged = dict1.copy()
        merged.update(**kwargs)
        return merged
    
    
    read_variables = read_variables_from_namelist(file_name)
    return merge_variables(read_variables)

    


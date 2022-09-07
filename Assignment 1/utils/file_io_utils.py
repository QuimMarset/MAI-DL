import os
import json
from textwrap import indent



def write_array_to_file(array, file_path):
    with open(file_path, 'w') as file:
        for element in array:
            file.write(element)


def read_file_to_array(file_path):
    array = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            array.append(line)
    return array


def write_json_string(json_string, file_path):
    with open(file_path, 'w') as file:
        json.dump(json.loads(json_string), file, indent=4)


def write_dict_to_json(dict, file_path, indent=4):
    with open(file_path, 'w') as file:
        json.dump(dict, file, indent=indent)


def load_json_to_dict(load_path):
    with open(load_path, 'r') as file:
        dict = json.load(file)
    return dict

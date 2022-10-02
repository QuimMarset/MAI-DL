import numpy as np
import json



def write_json_string(json_string, file_path):
    with open(file_path, 'w') as file:
        json.dump(json.loads(json_string), file, indent=4)


def write_dict_to_json(dict, file_path, indent=4):
    with open(file_path, 'w') as file:
        json.dump(dict, file, indent=indent)


def load_json_to_dict(file_path):
    with open(file_path, 'r') as file:
        dict = json.load(file)
    return dict


def save_mean_std_to_npy(mean, std, file_path):
    with open(file_path, 'wb') as file:
        np.save(file, mean)
        np.save(file, std)


def load_mean_std_from_npy(file_path):
    with open(file_path, 'rb') as file:
        mean = np.load(file)
        std = np.load(file)
    return mean, std
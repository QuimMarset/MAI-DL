import os


def create_folder(path, folder_name):
    path = os.path.join(path, folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_number_subfolders(path):
    return sum(os.path.isdir(elem) for elem in os.listdir(path))


def join_path(path, name):
    return os.path.join(path, name)
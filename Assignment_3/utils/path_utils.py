import os



def join_path(base_path, *path_names):
    return os.path.join(base_path, *path_names)


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def get_subfolders_paths(path):
    subfolders = []
    for element in sorted(os.listdir(path)):
        sub_path = join_path(path, element)
        if os.path.isdir(sub_path):
            subfolders.append(sub_path)
    return subfolders


def sort_experiments_paths(experiments_paths):
    return sorted(experiments_paths, key=lambda path: int(path.split('_')[-1]))


def get_exercise_experiments_paths(exercise_path):
    experiments_paths = get_subfolders_paths(exercise_path)
    return sort_experiments_paths(experiments_paths)



def get_number_subfolders(path):
    return sum(os.path.isdir(join_path(path, elem)) for elem in os.listdir(path))


def create_new_experiment_folder(path):
    num_experiments = get_number_subfolders(path)
    experiment_path = join_path(path, f'experiment_{num_experiments+1}')
    create_folder(experiment_path)
    return experiment_path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.path_utils import join_path, get_exercise_experiments_paths
from utils.file_io_utils import load_json_to_dict


def create_y_ticks():
    ticks = [pow(10, exp) for exp in range(-15, 2)]
    return ticks


def plot_loss_curves_log_scale(exercise_path, exercise_num, x_label):
    losses_dict = prepare_metric_values(exercise_path, 'loss')
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(losses_dict)))

    for i, experiment in enumerate(losses_dict):
        plt.plot(losses_dict[experiment], label=experiment, color=colors[i])

    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.yticks(create_y_ticks())
    plt.legend()
    plt.title(f'Exercise {exercise_num}: Loss Curves Comparison')
    plt.tight_layout()
    plt.savefig(join_path(exercise_path, 'loss_curves.png'), dpi=200)
    plt.close()


def plot_loss_curves(exercise_path, exercise_num, x_label):
    losses_dict = prepare_metric_values(exercise_path, 'loss')
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(losses_dict)))

    for i, experiment in enumerate(losses_dict):
        plt.plot(losses_dict[experiment][300:], label=experiment, color=colors[i])

    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Exercise {exercise_num}: Loss Curves Comparison')
    plt.tight_layout()
    plt.savefig(join_path(exercise_path, 'loss_curves.png'), dpi=300)
    plt.close()


def plot_test_accuracy_curves(exercise_path, exercise_num, x_label):
    losses_dict = prepare_metric_values(exercise_path, 'test_accuracy')
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(losses_dict)))

    for i, experiment in enumerate(losses_dict):
        plt.plot(losses_dict[experiment][300:], label=experiment, color=colors[i])

    plt.xlabel(x_label)
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title(f'Exercise {exercise_num}: Test Accuracy Curves Comparison')
    plt.tight_layout()
    plt.savefig(join_path(exercise_path, 'test_accuracy_curves.png'), dpi=300)
    plt.close()


def prepare_metric_values(exercise_path, metric_name):
    experiments_paths = get_exercise_experiments_paths(exercise_path)
    losses_dict = {}
    for i, experiment_path in enumerate(experiments_paths):
        losses_i = load_json_to_dict(join_path(experiment_path, 'metrics.json'))[metric_name]
        experiment_name = f'Experiment {i+1}'
        losses_dict[experiment_name] = losses_i
    return losses_dict
from utils.file_io_utils import load_json_to_dict
import seaborn as sns
import matplotlib.pyplot as plt
from utils.path_utils import join_path


def plot_values(values):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.plot(values, label='Gradients')

    plt.xlabel('Iteration')
    plt.ylabel('Gradient')
    plt.yscale('symlog')
    plt.legend()
    plt.title(f'Exercise 1: Experiment 11 Gradient Curve')
    plt.tight_layout()
    plt.savefig(join_path('./', 'gradient_curve.png'), dpi=200)
    plt.close()


if __name__ == '__main__':
    path = 'experiments/exercise_1/experiment_11/metrics.json'

    dict = load_json_to_dict(path)

    gradients = dict['gradient']
    plot_values(gradients[200:370])



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.paths_utils import join_path
from utils.image_utils import load_image



def plot_learning_curve(train_values, val_values, metric_name, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 6))

    num_epochs = len(train_values)
    epochs_range = range(1, num_epochs+1)

    plt.plot(epochs_range, train_values, label=f'Train {metric_name}')
    plt.plot(epochs_range, val_values, label=f'Val {metric_name}')
    
    plt.xticks(epochs_range)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Training and validation {metric_name} curves')
    plt.tight_layout()
    plt.savefig(join_path(save_path, f'{metric_name}_curves.png'))
    plt.close()


def compute_class_frequencies(labels, num_classes):
    frequencies = np.zeros(num_classes, dtype=float)
    for label in labels:
        frequencies[label] += 1
    frequencies /= np.sum(frequencies)
    return frequencies


def plot_classes_histogram(labels, class_names, save_path, partition='whole', log_scale=False):
    sns.set(style="whitegrid")
    num_classes = len(class_names)
    frequencies = compute_class_frequencies(labels, num_classes)
    plt.figure(figsize=(8, 6))
    plt.bar(range(num_classes), frequencies, label='Class proportion', log=log_scale)
    plt.xticks([i for i in range(num_classes)], class_names, rotation='vertical')
    plt.legend()
    plt.title(f'Classes proportion in the {partition} dataset')
    plt.tight_layout()
    plt.savefig(join_path(save_path, f'{partition}_class_proportion.png'))
    plt.close()


def plot_random_25_images(images_path, image_names, labels, class_names):
    plt.figure(figsize=(10, 10))

    indices_range = range(len(image_names))
    selected_indices = np.random.choice(indices_range, 25, replace=False)
    
    for i, selected_index in enumerate(selected_indices):
        plt.subplot(5, 5, i+1)
        plt.imshow(load_image(images_path, image_names[selected_index]))
        label_num = labels[selected_index]
        plt.title(class_names[label_num], fontsize=10)
        plt.axis('off')

    plt.tight_layout()     
    plt.show()

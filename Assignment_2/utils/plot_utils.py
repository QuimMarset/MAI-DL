import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from utils.path_utils import join_path



def plot_learning_curve(train_values, val_values, metric_name, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    num_epochs = len(train_values)
    epochs_range = range(1, num_epochs+1)

    plt.plot(epochs_range, train_values, label=f'Train {metric_name}')
    plt.plot(epochs_range, val_values, label=f'Val {metric_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Training and validation {metric_name} curves')
    plt.tight_layout()
    plt.savefig(join_path(save_path, f'{metric_name}_curves.png'))
    plt.close()


def plot_learning_curves(train_loss, val_loss, train_acc, val_acc, save_path):
    sns.set(style="whitegrid")
    _, ax = plt.subplots(figsize=(12, 6))

    ax.plot(train_loss, 'g-.', label='Train loss')
    ax.plot(val_loss, 'r-.', label='Val loss')
    ax.set_ylabel('Loss')
    
    ax_2 = ax.twinx()
    ax_2.plot(train_acc, 'c', label='Train accuracy')
    ax_2.plot(val_acc, color='orange', label='Val accuracy')
    ax_2.set_ylabel('Accuracy')

    ax.set_xlabel('Epoch')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, bbox_to_anchor=(1.08, 0.5), loc="center left", fontsize=9)
    plt.title(f'Training and validation learning curves')
    plt.tight_layout()
    plt.savefig(join_path(save_path, 'learning_curves.png'), dpi=100)
    plt.close()


def compute_class_frequencies(labels, class_index_dict, num_classes):
    frequencies = np.zeros(num_classes, dtype=float)
    for label in labels:
        index = class_index_dict[label]
        frequencies[index] += 1
    frequencies /= np.sum(frequencies)
    return frequencies


def plot_classes_histogram(labels, class_names, class_index_dict, save_path, partition='whole'):
    sns.set(style="whitegrid")
    num_classes = len(class_names)
    frequencies = compute_class_frequencies(labels, class_index_dict, num_classes)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(num_classes), frequencies, label='Class proportion')
    plt.xticks([i for i in range(num_classes)], class_names, rotation='vertical')
    plt.legend()
    plt.title(f'Classes proportion in the {partition} dataset')
    plt.tight_layout()
    plt.savefig(join_path(save_path, f'{partition}_class_proportion.png'))
    plt.close()


def plot_random_25_images(images_paths, labels):

    def load_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    plt.figure(figsize=(10, 10))

    indices_range = range(len(images_paths))
    selected_indices = np.random.choice(indices_range, 25, replace=False)
    
    for i, selected_index in enumerate(selected_indices):
        plt.subplot(5, 5, i+1)
        plt.imshow(load_image(images_paths[selected_index]))
        plt.title(labels[selected_index], fontsize=10)
        plt.axis('off')

    plt.tight_layout()     
    plt.show()


def plot_confusion_matrix(values, class_names, save_path):
    confusion_matrix_df = pd.DataFrame(values, index=class_names, columns=class_names)
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 16}, fmt='g')
    plt.tight_layout()
    plt.savefig(join_path(save_path, f'test_confusion_matrix.png'), dpi=300)
    plt.close()
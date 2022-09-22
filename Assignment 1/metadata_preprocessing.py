import pandas as pd
import os
import numpy as np
from utils.paths_utils import create_folder, join_path
from utils.plot_utils import plot_classes_histogram
from utils.file_io_utils import *



def preprocess_labels_metadata(labels_csv_path, save_path):
    labels_dataframe = pd.read_csv(labels_csv_path, header=None)
    # Col 1 contains the class names, and col 0 the indices (0...28)
    labels_dict = dict(zip(labels_dataframe[1], labels_dataframe[0]))
    write_string_array_to_file(list(labels_dict.keys()), join_path(save_path, 'class_names.txt'))
    return labels_dict


def write_names_labels_to_file(partition_dataframe, label_dict, save_path, partition_name):
    names = partition_dataframe['Image file'].to_numpy()
    labels =  partition_dataframe['Medium'].to_numpy()
    encoded_labels = [label_dict[label.strip()] for label in labels]
    write_string_array_to_file(names, join_path(save_path, f'{partition_name}_data.txt'))
    write_int_array_to_file(encoded_labels, join_path(save_path, f'{partition_name}_labels.txt'))


def preprocess_dataset_metadata(dataset_csv_path, labels_dict, save_path):
    dataframe = pd.read_csv(dataset_csv_path)
    train_dataframe = dataframe[dataframe['Subset'] == 'train']
    val_dataframe = dataframe[dataframe['Subset'] == 'val']
    test_dataframe = dataframe[dataframe['Subset'] == 'test']
    write_names_labels_to_file(train_dataframe, labels_dict, save_path, 'train')
    write_names_labels_to_file(val_dataframe, labels_dict, save_path, 'val')
    write_names_labels_to_file(test_dataframe, labels_dict, save_path, 'test')




if __name__ == '__main__':

    root_path = './'
    metadata_path = join_path(root_path, 'MAMe_metadata')
    dataset_csv_path = join_path(metadata_path, 'MAMe_dataset.csv')
    class_def_csv_path = join_path(metadata_path, 'MAMe_labels.csv')

    processed_folder = create_folder(root_path, 'processed_dataset')
    splits_folder = create_folder(processed_folder, 'dataset_splits')
    statistics_folder = create_folder(processed_folder, 'dataset_statistics')

    # Extract the class names and save them in a txt. Returns the dictionary to map names to indices (same order as in csv)
    labels_dict = preprocess_labels_metadata(class_def_csv_path, splits_folder)
    # Extract from the csv the images and labels for each split (train/val/test) and save them in txt files
    preprocess_dataset_metadata(dataset_csv_path, labels_dict, splits_folder)

    # The keys in the dictionary as in the same order as in the csv
    class_names = list(labels_dict.keys())

    train_labels = read_file_to_int_array(join_path(splits_folder, 'train_labels.txt'))
    val_labels = read_file_to_int_array(join_path(splits_folder, 'val_labels.txt'))
    test_labels = read_file_to_int_array(join_path(splits_folder, 'test_labels.txt'))
    full_labels = np.concatenate([train_labels, val_labels, test_labels])

    # Plot the class frequencies for the whole dataset and for each partition to see the class balance
    plot_classes_histogram(full_labels, class_names, statistics_folder, 'whole')
    plot_classes_histogram(train_labels, class_names, statistics_folder, 'train')
    plot_classes_histogram(val_labels, class_names, statistics_folder, 'val')
    plot_classes_histogram(test_labels, class_names, statistics_folder, 'test')
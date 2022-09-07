import pandas as pd
import os
from utils.file_io_utils import *



def write_names_labels_to_file(partition_dataframe, label_dict, save_path, partition_name):
    names_array = partition_dataframe['Image file'].to_numpy()
    labels_array =  partition_dataframe['Medium'].to_numpy()
    encoded_labels_list = [label_dict[label] for label in labels_array]

    write_array_to_file(names_array, os.path.join(save_path, f'{partition_name}_data.txt'))
    write_array_to_file(encoded_labels_list, os.path.join(save_path, f'{partition_name}_labels.txt'))


def create_label_dict(labels_csv_path):
    dataframe = pd.read_csv(labels_csv_path, header=None)
    return dict(zip(dataframe[1], dataframe[0]))


def preprocess_metadata(dataset_csv_path, labels_csv_path, save_path):
    dataframe = pd.read_csv(dataset_csv_path)
    labels_dict = create_label_dict(labels_csv_path)
    
    train_dataframe = dataframe[dataframe['Subset'] == 'train']
    val_dataframe = dataframe[dataframe['Subset'] == 'val']
    test_dataframe = dataframe[dataframe['Subset'] == 'test']

    write_names_labels_to_file(train_dataframe, labels_dict, save_path, 'train')
    write_names_labels_to_file(val_dataframe, labels_dict, save_path, 'val')
    write_names_labels_to_file(test_dataframe, labels_dict, save_path, 'test')
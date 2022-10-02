import pandas as pd
from utils.paths_utils import join_path


def create_subset_dataframe(dataframe, data_path, subset_name):
    image_files = dataframe[dataframe['Subset'] == subset_name]['Image file'].tolist()
    labels = dataframe[dataframe['Subset'] == subset_name]['Medium'].tolist()
    dataframe = pd.DataFrame({'filename': image_files, 'class': labels})
    dataframe['filename'] = dataframe['filename'].transform(lambda filename: join_path(data_path, filename))
    dataframe['class'] = dataframe['class'].transform(lambda class_name: class_name.strip())
    return dataframe


def create_train_val_dataframes(data_csv_path, data_path):
    dataframe = pd.read_csv(data_csv_path)
    train_df = create_subset_dataframe(dataframe, data_path, 'train')
    val_df = create_subset_dataframe(dataframe, data_path, 'val')
    return train_df, val_df


def create_test_dataframe(data_csv_path, data_path):
    dataframe = pd.read_csv(data_csv_path)
    return create_subset_dataframe(dataframe, data_path, 'test')


def get_classes_names(classes_csv_path):
    return pd.read_csv(classes_csv_path, header=None)[1].to_numpy()
import numpy as np
from utils.paths_utils import create_folder, join_path
from utils.plot_utils import plot_classes_histogram
from utils.csv_utils import create_train_val_dataframes, create_test_dataframe, get_classes_names


if __name__ == '__main__':

    root_path = './'
    data_path = join_path(root_path, 'data_256')
    data_csv_path = join_path(root_path, 'MAMe_metadata', 'MAMe_dataset.csv')
    data_statistics_path = create_folder(root_path, 'data_statistics')
    
    train_df, val_df = create_train_val_dataframes(data_csv_path, data_path)
    test_df = create_test_dataframe(data_csv_path, data_path)

    train_labels = train_df['class'].to_numpy()
    val_labels = val_df['class'].to_numpy()
    test_labels = test_df['class'].to_numpy()
    full_labels = np.concatenate([train_labels, val_labels, test_labels])

    class_names = get_classes_names(join_path(root_path, 'MAMe_metadata', 'MAMe_labels.csv'))
    class_index_dict = dict(zip(class_names, range(len(class_names))))

    # Plot the class frequencies for the whole dataset and for each partition to see the class balance
    plot_classes_histogram(full_labels, class_names, class_index_dict, data_statistics_path, 'whole')
    plot_classes_histogram(train_labels, class_names, class_index_dict, data_statistics_path, 'train')
    plot_classes_histogram(val_labels, class_names, class_index_dict, data_statistics_path, 'val')
    plot_classes_histogram(test_labels, class_names, class_index_dict, data_statistics_path, 'test')
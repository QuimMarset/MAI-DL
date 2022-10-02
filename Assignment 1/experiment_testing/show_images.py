import sys
sys.path.append('./')

from utils.paths_utils import *
from utils.plot_utils import plot_random_25_images
from utils.csv_utils import create_train_val_dataframes, get_classes_names



if __name__ == '__main__':

    root_path = './'
    data_path = join_path(root_path, 'data_256')
    data_csv_path = join_path(root_path, 'MAMe_metadata', 'MAMe_dataset.csv')
    data_statistics_path = create_folder(root_path, 'data_statistics')
    
    train_df, _ = create_train_val_dataframes(data_csv_path, data_path)
    train_paths = train_df['filename'].to_numpy()
    train_labels = train_df['class'].to_numpy()

    plot_random_25_images(train_paths, train_labels)
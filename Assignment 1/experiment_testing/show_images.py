from utils.paths_utils import *
from utils.plot_utils import plot_random_25_images
from utils.file_io_utils import read_file_to_array, read_file_to_int_array



if __name__ == '__main__':

    root_path = './'
    dataset_path = join_path(root_path, 'data_256')
    splits_path = join_path(root_path, 'processed_dataset', 'dataset_splits')

    train_names = read_file_to_array(join_path(splits_path, 'train_data.txt'))
    train_labels = read_file_to_int_array(join_path(splits_path, 'train_labels.txt'))
    class_names = read_file_to_array(join_path(splits_path, 'class_names.txt'))

    plot_random_25_images(dataset_path, train_names, train_labels, class_names)
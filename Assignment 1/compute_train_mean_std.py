import numpy as np
from utils.paths_utils import *
from utils.file_io_utils import read_file_to_array, save_mean_std_to_npy
from utils.image_utils import *



if __name__ == '__main__':

    root_path = './'
    processed_path = join_path(root_path, 'processed_dataset')
    data_path = join_path(root_path, 'data_256')
    splits_path = join_path(processed_path, 'dataset_splits')
    train_split = join_path(splits_path, 'train_data.txt')

    train_names = read_file_to_array(train_split)

    per_channel_sum = np.zeros(3)
    per_channel_squared_sum = np.zeros(3)

    for (i, name) in enumerate(train_names):
        image = load_image(data_path, name)
        image = image_to_float(image)

        per_channel_sum += np.sum(image, axis=(0, 1))
        per_channel_squared_sum += np.sum(image**2, axis=(0, 1))

    
    count = len(train_names) * 256 * 256 
    mean = per_channel_sum / count
    var = (per_channel_squared_sum / count) - (mean ** 2)
    std = np.sqrt(var)

    print(f'Mean and std from the training images: \nMean: {mean}\nStd: {std}')

    npy_path = join_path(processed_path, 'train_mean_std.npy')
    save_mean_std_to_npy(mean, std, npy_path)



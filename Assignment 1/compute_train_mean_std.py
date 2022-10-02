import numpy as np
from utils.paths_utils import *
from utils.file_io_utils import save_mean_std_to_npy
from utils.csv_utils import create_train_val_dataframes
from utils.image_utils import *



if __name__ == '__main__':

    root_path = './'
    data_path = join_path(root_path, 'data_256')
    data_csv_path = join_path(root_path, 'MAMe_metadata', 'MAMe_dataset.csv')
    data_statistics_path = create_folder(root_path, 'data_statistics')
    
    train_df, _ = create_train_val_dataframes(data_csv_path, data_path)
    train_paths = train_df['filename'].to_numpy()
    
    per_channel_sum = np.zeros(3)
    per_channel_squared_sum = np.zeros(3)

    for (i, image_path) in enumerate(train_paths):
        image = load_image(image_path)
        image = image_to_float(image)

        per_channel_sum += np.sum(image, axis=(0, 1))
        per_channel_squared_sum += np.sum(image**2, axis=(0, 1))

    
    count = len(train_paths) * 256 * 256 
    mean = per_channel_sum / count
    var = (per_channel_squared_sum / count) - (mean ** 2)
    std = np.sqrt(var)

    print(f'Mean and std from the training images: \nMean: {mean}\nStd: {std}')

    npy_path = join_path(data_statistics_path, 'train_mean_std.npy')
    save_mean_std_to_npy(mean, std, npy_path)



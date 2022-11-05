from utils.path_utils import join_path


data_path = 'data_256'

metadata_path = 'MAMe_metadata'
data_csv_path = join_path(metadata_path, 'MAMe_dataset.csv')
classes_names_path = join_path(metadata_path, 'MAMe_labels.csv')
train_mean_std_path = join_path(metadata_path, 'train_mean_std.npy')

experiments_path = 'experiments'
experiments_fine_tune_path = join_path(experiments_path, 'fine_tune')
experiments_feature_extraction_path = join_path(experiments_path, 'feature_extraction')

test_model_path = 'test_model'

image_shape_no_top = (256, 256, 3)
image_shape_top = (224, 224, 3)
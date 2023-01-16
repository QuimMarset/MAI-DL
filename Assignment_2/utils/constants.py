from utils.path_utils import join_path


data_path = 'data_256'

metadata_path = 'MAMe_metadata'
data_csv_path = join_path(metadata_path, 'MAMe_dataset.csv')
classes_names_path = join_path(metadata_path, 'MAMe_labels.csv')
train_mean_std_path = join_path(metadata_path, 'train_mean_std.npy')

experiments_path = 'experiments'
experiments_fine_tune_path = join_path(experiments_path, 'fine_tune')
experiments_feature_extraction_path = join_path(experiments_path, 'feature_extraction')

test_fine_tune_model_path = join_path(experiments_fine_tune_path, 'experiment_10')
test_feat_ext_model_path = join_path(experiments_feature_extraction_path, 'experiment_17')

pre_trained_path = join_path('models', 'pre_trained_weights')
vgg_16_places_top_path = join_path(pre_trained_path, 'vgg16_places_top.h5')
vgg_16_places_no_top_path = join_path(pre_trained_path, 'vgg16_places_no_top.h5')
vgg_16_hybrid_top_path = join_path(pre_trained_path, 'vgg16_hybrid_top.h5')
vgg_16_hybrid_no_top_path = join_path(pre_trained_path, 'vgg16_hybrid_no_top.h5')


image_shape_no_top = (256, 256, 3)
image_shape_top = (224, 224, 3)
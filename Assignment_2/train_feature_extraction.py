from sklearn.metrics import accuracy_score
from utils.path_utils import *
from utils.csv_utils import *
from utils.file_io_utils import *
from utils.constants import *
from image_data_generator import *
from models.feature_extraction_model import FeatureExtractionModel


def get_train_labels(train_gen):
    labels = []
    for _ in range(len(train_gen)):
        _, batch_labels = next(train_gen)
        labels.extend(batch_labels)
    return np.array(labels, dtype=int)


def train(experiment_path, config, train_gen, val_gen, num_classes, seed):
    model = FeatureExtractionModel(config.name, config.feature_layers, num_classes, 
        config.standarization, config.discretization, config.neg_threshold, config.pos_threshold, seed)

    train_labels = get_train_labels(train_gen)
    train_predictions, val_predictions = model.train_classifier(train_gen, train_labels, val_gen)
    model.save_model(experiment_path)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    val_accuracy = accuracy_score(val_gen.labels, val_predictions)

    training_metrics = {'train_predictions' : train_predictions.tolist(), 'val_predictions' : val_predictions.tolist(),
        'train_accuracy' : train_accuracy, 'val_accuracy' : val_accuracy}
    write_dict_to_json(training_metrics, join_path(experiment_path, 'train_metrics.json'))
    
    hyperparameters = vars(config)
    write_dict_to_json(hyperparameters, join_path(experiment_path, 'hyperparameters.json'))



if __name__ == '__main__':
    os.environ['TF_DETERMINISTIC_OPS']  = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    config = read_yaml_config('config_feature_extraction.yaml')
    
    create_folder(experiments_feature_extraction_path)
    experiment_path = create_new_experiment_folder(experiments_feature_extraction_path)

    train_mean_std = load_mean_std_from_npy(train_mean_std_path)
    classes_names = get_classes_names(classes_names_path)
    
    train_df, val_df = create_train_val_dataframes(data_csv_path, data_path)

    batch_size = config.batch_size
    seed = config.seed
    use_augmentation = config.augmentation
    image_shape = image_shape_top

    train_gen = create_train_generator(train_df, image_shape[:-1], batch_size, use_augmentation, config.normalization, config.name, seed, 'sparse')
    val_gen = create_val_generator(val_df, image_shape[:-1], batch_size, config.normalization, config.name, 'sparse')

    train(experiment_path, config, train_gen, val_gen, len(classes_names), seed)
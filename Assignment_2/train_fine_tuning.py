from tensorflow import keras
from utils.path_utils import *
from utils.csv_utils import *
from utils.plot_utils import *
from utils.file_io_utils import *
from utils.constants import *
from image_data_generator import *
from models.fine_tuning_model import FineTuningModel



def define_learning_rate(config, epoch_steps):
    if config.lr_decay:
        return keras.optimizers.schedules.ExponentialDecay(config.lr, 
            config.decay_epochs*epoch_steps, config.decay_rate)
    else:
        return config.lr


def create_optimizer(config, epoch_steps):
    learning_rate = define_learning_rate(config, epoch_steps)
    return keras.optimizers.Adam(learning_rate)


def create_model(config, image_shape, num_classes, seed):
    return FineTuningModel(config.name, image_shape, config.num_to_freeze, 
        config.dense_list, config.batch_norm, config.dropout, config.activation, 
        num_classes, seed)


def train(experiment_path, config, train_gen, val_gen, image_shape, num_classes, seed):
    optimizer = create_optimizer(config, len(train_gen))
    
    model = create_model(config, image_shape, num_classes, seed)
    model.compile(optimizer, config.label_smoothing)
    history = model.fit(train_gen, config.epochs, val_gen, config.patience)
    model.save_model(experiment_path)
    
    training_metrics = history.history
    write_dict_to_json(training_metrics, join_path(experiment_path, 'train_metrics.json'))
    
    hyperparameters = vars(config)
    write_dict_to_json(hyperparameters, join_path(experiment_path, 'hyperparameters.json'))

    plot_learning_curves(training_metrics['loss'], training_metrics['val_loss'], 
        training_metrics['accuracy'], training_metrics['val_accuracy'], experiment_path)



if __name__ == '__main__':
    os.environ['TF_DETERMINISTIC_OPS']  = '1'

    config = read_yaml_config('config_fine_tuning.yaml')
    
    create_folder(experiments_fine_tune_path)
    experiment_path = create_new_experiment_folder(experiments_fine_tune_path)

    train_mean_std = load_mean_std_from_npy(train_mean_std_path)
    classes_names = get_classes_names(classes_names_path)
    
    mean_std = None
    if config.normalization:
        mean_std = train_mean_std
    
    train_df, val_df = create_train_val_dataframes(data_csv_path, data_path)

    batch_size = config.batch_size
    seed = config.seed
    use_augmentation = config.augmentation

    train_gen = create_train_generator(train_df, image_shape[:-1], batch_size, use_augmentation, mean_std, seed)
    val_gen = create_val_generator(val_df, image_shape[:-1], batch_size, mean_std)

    train(experiment_path, config, train_gen, val_gen, image_shape, len(classes_names), seed)
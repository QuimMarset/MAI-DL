from tensorflow import keras
import tensorflow_addons as tfa
import random
from utils.paths_utils import *
from utils.csv_utils import create_train_val_dataframes, get_classes_names
from utils.plot_utils import *
from utils.argument_parser import parse_input_arguments
from utils.file_io_utils import write_dict_to_json, load_mean_std_from_npy
from image_data_generator import create_train_generator, create_val_generator
from models import cnn, cnn_residuals


def generate_tf_random_seeds(args):
    global_seed = args.global_seed if args.seed else random.randint(0, 9999)
    operational_seed = args.operational_seed if args.seed else random.randint(0, 9999)
    return global_seed, operational_seed


def define_learning_rate(args, epoch_steps):
    if args.lr_decay:
        return keras.optimizers.schedules.ExponentialDecay(args.lr, args.decay_epochs*epoch_steps, args.decay_rate)
    else:
        return args.lr


def create_optimizer(args, epoch_steps):
    learning_rate = define_learning_rate(args, epoch_steps)

    if args.optimizer == 'SGD':
        return keras.optimizers.SGD(learning_rate)
    else:
        if args.weight_decay > 0:
            pass
            return tfa.optimizers.AdamW(weight_decay=args.weight_decay, learning_rate=learning_rate)
        else:
            return keras.optimizers.Adam(learning_rate)


def create_model(args, input_shape, num_classes, global_seed, operational_seed):
    use_max_pooling = args.pooling_type == 'Max'
    kernel_sizes = args.kernel_sizes * len(args.conv_filters) if len(args.kernel_sizes) == 1 else args.kernel_sizes

    if args.model == 'CNN':
        return cnn.CNNWrapper(input_shape, kernel_sizes, args.conv_filters, 
            args.padding, args.dense_units, args.batch_norm, args.dropout, use_max_pooling, 
            args.activation, num_classes, global_seed, operational_seed)
    else:
        return cnn_residuals.CNNResidualsWrapper(input_shape, kernel_sizes, args.conv_filters, 
            args.dense_units, args.batch_norm, args.dropout, use_max_pooling, args.activation, 
            num_classes, global_seed, operational_seed)


def define_callbacks(parsed_args):
    callbacks = []

    if parsed_args.early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=parsed_args.patience, 
            restore_best_weights=True, min_delta=parsed_args.min_delta)
        
        callbacks.append(early_stopping)

    return callbacks


def train(experiment_path, parsed_args, train_gen, val_gen, image_shape, num_classes, global_seed, operational_seed):
    optimizer = create_optimizer(parsed_args, len(train_gen))
    
    model_wrapper = create_model(parsed_args, image_shape, num_classes, global_seed, operational_seed)
    model_wrapper.save_architecture(experiment_path)
    model_wrapper.save_summary(experiment_path)

    callbacks = define_callbacks(parsed_args)

    loss_function = keras.losses.CategoricalCrossentropy(label_smoothing=parsed_args.label_smoothing)
    model_wrapper.model.compile(optimizer, run_eagerly=True, loss=loss_function, metrics=["accuracy"])
    history = model_wrapper.model.fit(train_gen, epochs=parsed_args.epochs, validation_data=val_gen, workers=6, callbacks=callbacks)

    model_wrapper.save_model_weights(experiment_path)
    
    # Save training metrics and experiment hyperparameters
    training_metrics = history.history
    write_dict_to_json(training_metrics, join_path(experiment_path, 'train_metrics.json'))
    
    hyperparameters = vars(parsed_args)
    hyperparameters['global_seed'] = global_seed
    hyperparameters['operational_seed'] = operational_seed
    write_dict_to_json(hyperparameters, join_path(experiment_path, 'hyperparameters.json'))

    plot_learning_curves(training_metrics['loss'], training_metrics['val_loss'], 
        training_metrics['accuracy'], training_metrics['val_accuracy'], experiment_path)



if __name__ == '__main__':

    parsed_args = parse_input_arguments()
    
    root_path = './'
    data_path = join_path(root_path, 'data_256')
    data_csv_path = join_path(root_path, 'MAMe_metadata', 'MAMe_dataset.csv')
    data_statistics_path = join_path(root_path, 'data_statistics')
    experiments_path = create_folder(root_path, 'experiments')
    new_experiment_path = create_new_experiment_folder(experiments_path)

    train_mean_std = load_mean_std_from_npy(join_path(data_statistics_path, 'train_mean_std.npy'))
    classes_names = get_classes_names(join_path(root_path, 'MAMe_metadata', 'MAMe_labels.csv'))
    image_shape = (256, 256, 3)
    
    global_seed, operational_seed = generate_tf_random_seeds(parsed_args)

    mean_std = None #TODO: Add option to use instead of only 1/255
    
    train_df, val_df = create_train_val_dataframes(data_csv_path, data_path)
    train_gen = create_train_generator(train_df, image_shape[:-1], parsed_args.batch_size, parsed_args.augmentation, mean_std, operational_seed)
    val_gen = create_val_generator(val_df, image_shape[:-1], parsed_args.batch_size, mean_std, operational_seed)

    train(new_experiment_path, parsed_args, train_gen, val_gen, image_shape, len(classes_names), global_seed, operational_seed)

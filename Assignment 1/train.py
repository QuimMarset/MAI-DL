from tensorflow import keras
import tensorflow_addons as tfa
from utils.paths_utils import *
from utils.file_io_utils import *
from utils.plot_utils import *
from utils.argument_parser import parse_input_arguments
from data_generator import DataGenerator, DataGeneratorAugmentation
from models.cnn import CNNWrapper



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


def create_model(args, input_shape, num_classes):
    use_max_pooling = args.pooling_type == 'Max'
    kernel_sizes = args.kernel_sizes * len(args.conv_filters) if len(args.kernel_sizes) == 1 else args.kernel_sizes

    return CNNWrapper(input_shape, kernel_sizes, args.conv_filters, args.padding, args.dense_units, args.batch_norm,
        args.dropout, use_max_pooling, args.activation, num_classes)


def define_callbacks(parsed_args):
    callbacks = []

    if parsed_args.early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=parsed_args.patience, 
            restore_best_weights=True, min_delta=parsed_args.min_delta)
        
        callbacks.append(early_stopping)

    return callbacks


def train(experiment_path, parsed_args, train_gen, val_gen, image_shape, num_classes):
    optimizer = create_optimizer(parsed_args, len(train_gen))
    
    model_wrapper = create_model(parsed_args, image_shape, num_classes)
    model_wrapper.save_architecture(experiment_path)
    model_wrapper.save_summary(experiment_path)

    callbacks = define_callbacks(parsed_args)
    
    model_wrapper.model.compile(optimizer, run_eagerly=True, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model_wrapper.model.fit(train_gen, epochs=parsed_args.epochs, validation_data=val_gen, workers=6, callbacks=callbacks)
    
    model_wrapper.save_model_weights(experiment_path)
    # Save training metrics and experiment hyperparameters
    training_metrics = history.history
    write_dict_to_json(training_metrics, join_path(experiment_path, 'train_metrics.json'))
    write_dict_to_json(vars(parsed_args), join_path(experiment_path, 'hyperparameters.json'))

    plot_learning_curve(training_metrics['loss'], training_metrics['val_loss'], 'loss', experiment_path)
    plot_learning_curve(training_metrics['accuracy'], training_metrics['val_accuracy'], 'accuracy', experiment_path)



if __name__ == '__main__':

    root_path = './'
    dataset_path = join_path(root_path, 'data_256')
    processed_data_path = join_path(root_path, 'processed_dataset')
    splits_path = join_path(processed_data_path, 'dataset_splits')
    
    experiments_path = create_folder(root_path, 'experiments')
    new_experiment_path = create_new_experiment_folder(experiments_path)

    parsed_args = parse_input_arguments()

    train_names = read_file_to_array(join_path(splits_path, 'train_data.txt'))
    train_labels = read_file_to_int_array(join_path(splits_path, 'train_labels.txt'))
    val_names = read_file_to_array(join_path(splits_path, 'val_data.txt'))
    val_labels = read_file_to_int_array(join_path(splits_path, 'val_labels.txt'))

    train_mean_std = load_mean_std_from_npy(join_path(processed_data_path, 'train_mean_std.npy'))
    num_classes = len(read_file_to_array(join_path(splits_path, 'class_names.txt')))
    image_shape = (256, 256, 3)

    generator_class = DataGenerator
    if parsed_args.augmentation:
        generator_class = DataGeneratorAugmentation

    train_gen = generator_class(parsed_args.batch_size, dataset_path, train_names, image_shape, train_labels, num_classes, train_mean_std)
    val_gen = DataGenerator(parsed_args.batch_size, dataset_path, val_names, image_shape, val_labels, num_classes, train_mean_std)

    train(new_experiment_path, parsed_args, train_gen, val_gen, image_shape, num_classes)

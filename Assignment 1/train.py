from tensorflow import keras
from utils.paths_utils import *
from utils.file_io_utils import *
from utils.plot_utils import *
from utils.train_utils import *
from utils.experiments_utils import parse_input_arguments
from data_generator import DataGenerator, DataGeneratorAugmentation




def train(experiment_path, parsed_args, train_gen, val_gen, image_shape, num_classes):
    optimizer = create_optimizer(parsed_args, len(train_gen))
    
    model_wrapper = create_model(parsed_args, image_shape, num_classes)
    model_wrapper.save_architecture(experiment_path)
    model_wrapper.save_summary(experiment_path)
    
    callbacks = []

    if parsed_args.early_stop >= 0:
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=parsed_args.early_stop, restore_best_weights=True, min_delta=0.1)
        callbacks.append(early_stopping)

    model_wrapper.model.compile(optimizer, run_eagerly=True, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model_wrapper.model.fit(train_gen, epochs=parsed_args.epochs, validation_data=val_gen, workers=6)
    
    model_wrapper.save_model_weights(experiment_path)
    # Save training metrics and experiment hyperparameters
    training_metrics = history.history
    write_dict_to_json(training_metrics, join_path(experiment_path, 'train_metrics.json'))
    write_dict_to_json(vars(parsed_args), join_path(experiment_path, 'hyperparameters.json'))

    plot_learning_curve(training_metrics['loss'], training_metrics['val_loss'], 'loss', experiment_path)
    plot_learning_curve(training_metrics['accuracy'], training_metrics['val_accuracy'], 'accuracy', experiment_path)



if __name__ == '__main__':

    root_path = './'
    dataset_path = os.path.join(root_path, 'MAMe_data_256', 'data_256')
    processed_data_path = join_path(root_path, 'processed_dataset')
    splits_path = os.path.join(processed_data_path, 'dataset_splits')
    experiments_path = create_folder(root_path, 'experiments')
    new_experiment_path = create_new_experiment_folder(experiments_path)

    parsed_args = parse_input_arguments()

    train_names = read_file_to_array(join_path(splits_path, 'train_data.txt'))
    train_labels = read_file_to_int_array(join_path(splits_path, 'train_labels.txt'))
    val_names = read_file_to_array(join_path(splits_path, 'val_data.txt'))
    val_labels = read_file_to_int_array(join_path(splits_path, 'val_labels.txt'))

    train_mean_std_tuple = load_mean_std_from_npy(join_path(processed_data_path, 'train_mean_std.npy'))
    num_classes = len(read_file_to_array(join_path(splits_path, 'class_names.txt')))
    image_shape = (256, 256, 3)

    generator_class = DataGenerator
    if parsed_args.augmentation:
        generator_class = DataGeneratorAugmentation

    train_gen = generator_class(parsed_args.batch_size, dataset_path, train_names, image_shape, train_labels, num_classes, train_mean_std_tuple)
    val_gen = DataGenerator(parsed_args.batch_size, dataset_path, val_names, image_shape, val_labels, num_classes, train_mean_std_tuple)

    train(new_experiment_path, parsed_args, train_gen, val_gen, image_shape, num_classes)

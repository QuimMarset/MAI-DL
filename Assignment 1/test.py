from types import SimpleNamespace
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.paths_utils import *
from utils.csv_utils import create_test_dataframe, get_classes_names
from utils.plot_utils import plot_confusion_matrix
from utils.file_io_utils import load_mean_std_from_npy, load_json_to_dict
from image_data_generator import create_test_generator
from models import cnn, cnn_residuals



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


def test(parsed_args, weights_path, test_gen, image_shape, classes_name):
    model_wrapper = create_model(parsed_args, image_shape, len(classes_name), parsed_args.global_seed, parsed_args.operational_seed)
    model_wrapper.load_model_weights(weights_path)

    true_labels = test_gen.labels
    predictions = model_wrapper.model.predict(test_gen)
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    print(f'\nTest accuracy: {accuracy:.2f}\n')
    print(classification_report(true_labels, predictions, target_names=classes_name))
    confusion_matrix_ = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(confusion_matrix_, classes_names, weights_path)



if __name__ == '__main__':
    os.environ['TF_DETERMINISTIC_OPS']  = '1'
    
    root_path = './'
    weights_path = join_path(root_path, 'test_model')
    data_path = join_path(root_path, 'data_256')
    data_statistics_path = join_path(root_path, 'data_statistics')
    data_csv_path = join_path(root_path, 'MAMe_metadata', 'MAMe_dataset.csv')

    parameters = load_json_to_dict(join_path(weights_path, 'hyperparameters.json'))
    parameters = SimpleNamespace(**parameters)

    train_mean_std = load_mean_std_from_npy(join_path(data_statistics_path, 'train_mean_std.npy'))
    classes_names = get_classes_names(join_path(root_path, 'MAMe_metadata', 'MAMe_labels.csv'))
    image_shape = (256, 256, 3)
    
    mean_std = None
    if parameters.train_normalization:
        mean_std = train_mean_std

    test_df = create_test_dataframe(data_csv_path, data_path)
    test_gen = create_test_generator(test_df, image_shape[:-1], 32, mean_std)

    test(parameters, weights_path, test_gen, image_shape, classes_names)

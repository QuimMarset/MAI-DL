from utils.dataset_utils import load_mnist_dataset
from utils.optimizers_utils import create_optimizer
from utils.file_io_utils import write_dict_to_json, read_yaml_config
from utils.path_utils import join_path, create_folder, create_new_experiment_folder
from conv_network import ConvNetwork
from constants import config_3_path, exercise_3_experiments_path, mnist_path
import numpy as np



def create_model(config, image_shape, num_classes):
    return ConvNetwork(
        image_shape, config.kernel_size, config.conv_filters, config.dense_units,
        config.dropout, num_classes, config.l2_coef, config.seed
    )


def train(experiment_path, config, train_x, train_y, test_x, test_y, image_shape, num_classes):
    optimizer = create_optimizer(config)
    model = create_model(config, image_shape, num_classes)
    train_time, train_accuracies = model.train(train_x, train_y, config.batch_size, optimizer, config.gradient_clip, config.epochs)
    test_accuracy = model.test(test_x, test_y, config.batch_size)

    metrics = {
        'training_time': train_time,
        'training_accuracies' : train_accuracies,
        'test_accuracy' : test_accuracy
    }

    write_dict_to_json(metrics, join_path(experiment_path, 'metrics.json'))
    write_dict_to_json(vars(config), join_path(experiment_path, 'hyperparameters.json'))


def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


def compute_mean_std(train_x):
    mean = np.mean(train_x, axis=(0, 1, 2))
    std = np.std(train_x, axis=(0, 1, 2))
    return mean, std


def standarize_data(data, mean, std):
    return (data - mean) / (std + 1e-8)



if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    config = read_yaml_config(config_3_path)
    create_folder(exercise_3_experiments_path)
    experiment_path = create_new_experiment_folder(exercise_3_experiments_path)

    train_x, test_x, train_y_oh, test_y_oh = load_mnist_dataset(mnist_path, restore_images=True)

    train_y_oh = smooth_labels(train_y_oh, config.label_smoothing)

    #mean, std = compute_mean_std(train_x)
    #train_x = standarize_data(train_x, mean, std)
    #test_x = standarize_data(test_x, mean, std)

    image_shape = train_x.shape[1:]
    num_classes = train_y_oh.shape[1]

    train(experiment_path, config, train_x, train_y_oh, test_x, test_y_oh, image_shape, num_classes)


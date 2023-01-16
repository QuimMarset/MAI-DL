#!/usr/bin/env python
from utils.dataset_utils import load_mnist_dataset
from utils.optimizers_utils import create_optimizer
from utils.file_io_utils import read_yaml_config
from utils.path_utils import create_folder, create_new_experiment_folder
from conv_network_multi_gpu import ConvNetworkMultiGPU
from constants import config_4_path, exercise_4_experiments_path, mnist_path



def create_model(config, image_shape, num_classes):
    return ConvNetworkMultiGPU(
        image_shape, config.kernel_size, config.conv_filters, config.dense_units,
        config.dropout, num_classes, config.l2_coef, config.seed
    )


def train(experiment_path, config, train_x, train_y, test_x, test_y, image_shape, num_classes):
    optimizer = create_optimizer(config)
    model = create_model(config, image_shape, num_classes)
    model.compile(optimizer)
    model.train(train_x, train_y, config.batch_size, config.epochs)
    model.test(test_x, test_y, config.batch_size)



def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels



if __name__ == '__main__':

    config = read_yaml_config(config_4_path)
    create_folder(exercise_4_experiments_path)
    experiment_path = create_new_experiment_folder(exercise_4_experiments_path)

    train_x, test_x, train_y_oh, test_y_oh = load_mnist_dataset(mnist_path, restore_images=True)

    train_y_oh = smooth_labels(train_y_oh, config.label_smoothing)

    image_shape = train_x.shape[1:]
    num_classes = train_y_oh.shape[1]

    train(experiment_path, config, train_x, train_y_oh, test_x, test_y_oh, image_shape, num_classes)


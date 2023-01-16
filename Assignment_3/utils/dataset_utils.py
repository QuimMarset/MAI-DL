import pickle
import gzip
import numpy as np



def load_mnist_data_from_pickle(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    n_in = 28
    in_channel = 1
    num_classes = 10
   
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval, n_in, in_channel, num_classes


def labels_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[range(num_labels), labels] = 1
    return one_hot


def load_mnist_dataset(mnist_path, restore_images=False):
    data_input = load_mnist_data_from_pickle(mnist_path)
    data = data_input[0]
    side = data_input[1]
    num_channels = data_input[2]
    num_classes = data_input[3]

    train_x, train_y = data[0]
    test_x, test_y = data[2]

    if restore_images:
        train_x = np.reshape(train_x, [-1, side, side, num_channels])
        test_x = np.reshape(test_x, [-1, side, side, num_channels])

    train_y_one_hot = labels_to_one_hot(train_y, num_classes)
    test_y_one_hot = labels_to_one_hot(test_y, num_classes)

    return train_x, test_x, train_y_one_hot, test_y_one_hot
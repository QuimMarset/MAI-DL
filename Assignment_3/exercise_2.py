import tensorflow as tf
import numpy as np
from utils.dataset_utils import load_mnist_dataset
from utils.optimizers_utils import create_optimizer
from utils.file_io_utils import write_dict_to_json, read_yaml_config
from utils.path_utils import create_folder, create_new_experiment_folder, join_path
from utils.plot_utils import plot_loss_curves, plot_test_accuracy_curves
from constants import mnist_path, exercise_2_experiments_path, config_2_path



def compute_predictions(batch_x, W, b):
    return tf.nn.softmax(tf.matmul(batch_x, W) + b)


def compute_loss(batch_y, predictions):
    batch_y = tf.convert_to_tensor(batch_y, tf.float32)
    loss = tf.reduce_mean(-tf.reduce_sum(batch_y * tf.math.log(predictions), axis=1))
    return loss


def compute_test_accuracy(test_x, test_y, W, b):
    test_predictions = compute_predictions(test_x, W, b)
    test_predictions = tf.argmax(test_predictions, axis=1)
    test_y = np.argmax(test_y, axis=1)
    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(test_y, test_predictions), tf.float32))
    return test_accuracy.numpy()


def update_weights(batch_x, batch_y, W, b, optimizer, gradient_clip):    
    with tf.GradientTape() as tape:
        predictions = compute_predictions(batch_x, W, b)
        loss = compute_loss(batch_y, predictions)

    trainable_variables = [W, b]
    gradients = tape.gradient(loss, trainable_variables)

    if gradient_clip > 0:
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
    
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss.numpy()


def create_model_parameters(train_x, train_y):
    num_features = train_x.shape[-1]
    num_classes = train_y.shape[-1]
    W = tf.Variable(tf.zeros([num_features, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    return W, b


def train_optimizer(train_x, train_y, test_x, test_y, optimizer, num_batches, batch_size, gradient_clip):
    W, b = create_model_parameters(train_x, train_y)
    loss_sum = 0
    losses = []
    test_accuracies = []

    for i in range(num_batches):
        start = batch_size * i
        end = start + batch_size
        batch_x = train_x[start : end]
        batch_y = train_y[start : end]

        loss_i = update_weights(batch_x, batch_y, W, b, optimizer, gradient_clip)
        
        loss_sum += float(loss_i)
        current_loss_mean = loss_sum / (i + 1)
        losses.append(current_loss_mean)

        test_accuracy = compute_test_accuracy(test_x, test_y, W, b)
        test_accuracies.append(float(test_accuracy))
       
    return losses, test_accuracies


def train(train_x, train_y, config, test_x, test_y, experiment_path, num_batches, batch_size):
    optimizer = create_optimizer(config)
    losses, test_accuracies = train_optimizer(train_x, train_y, test_x, test_y, optimizer, 
        num_batches, batch_size, config.gradient_clip)
    write_dict_to_json(vars(config), join_path(experiment_path, 'hyperparameters.json'))

    metrics = {
        'loss' : losses,
        'test_accuracy' : test_accuracies,
    }
    write_dict_to_json(metrics, join_path(experiment_path, 'metrics.json'))



if __name__ == '__main__':

    #config = read_yaml_config(config_2_path)
#
    #create_folder(exercise_2_experiments_path)
    #experiment_path = create_new_experiment_folder(exercise_2_experiments_path)
#
    #train_x, test_x, train_y_oh, test_y_oh = load_mnist_dataset(mnist_path)
#
    #batch_size = 100
    #num_batches = train_x.shape[0] // batch_size
    #if train_x.shape[0] % batch_size != 0:
    #    num_batches += 1
#
    #train(train_x, train_y_oh, config, test_x, test_y_oh, experiment_path, num_batches, batch_size)
    plot_loss_curves(exercise_2_experiments_path, 2, 'Batch Number')
    plot_test_accuracy_curves(exercise_2_experiments_path, 2, 'Batch Number')
    
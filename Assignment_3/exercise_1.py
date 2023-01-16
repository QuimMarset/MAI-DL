import tensorflow as tf
import numpy as np
from utils.optimizers_utils import *
from utils.plot_utils import plot_loss_curves_log_scale
from utils.file_io_utils import read_yaml_config, write_dict_to_json
from utils.path_utils import create_folder, create_new_experiment_folder, join_path
from constants import exercise_1_experiments_path, config_1_path



def compute_loss(x_train, y_train, W, b):
    output = W * x_train + b
    loss = tf.reduce_sum(tf.square(output - y_train))
    return loss


def update_weights(x_train, y_train, W, b, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(x_train, y_train, W, b)

    trainable_variables = [W, b]
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    loss = compute_loss(x_train, y_train, W, b)
    return W, b, loss.numpy(), gradients[0].numpy()


def train_optimizer(x_train, y_train, optimizer, num_iterations):
    # Model parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    initial_loss = float(compute_loss(x_train, y_train, W, b).numpy())
    losses = [initial_loss]
    gradients = []
    Ws = [float(W.numpy()[0])]
    bs = [float(b.numpy()[0])]

    for _ in range(num_iterations):
        W, b, loss_i, gradient_i = update_weights(x_train, y_train, W, b, optimizer)
        losses.append(float(loss_i))
        gradients.append(float(gradient_i))
        Ws.append(float(W.numpy()[0]))
        bs.append(float(b.numpy()[0]))

    return losses, gradients, Ws, bs


def train(x_train, y_train, config, experiment_path, num_iterations=1000):
    optimizer = create_optimizer(config)
    losses, gradients, Ws, bs = train_optimizer(x_train, y_train, optimizer, num_iterations)
    write_dict_to_json(vars(config), join_path(experiment_path, 'hyperparameters.json'))

    metrics = {
        'loss' : losses,
        'gradient' : gradients,
        'W' : Ws,
        'b' : bs
    }

    write_dict_to_json(metrics, join_path(experiment_path, 'metrics.json'))
    


if __name__ == '__main__':

    #config = read_yaml_config(config_1_path)
    #create_folder(exercise_1_experiments_path)
    #experiment_path = create_new_experiment_folder(exercise_1_experiments_path)
#
    #x_train = np.array([1, 2, 3, 4])
    #y_train = np.array([0, -1, -2, -3])
#
    #train(x_train, y_train, config, experiment_path)
    plot_loss_curves_log_scale(exercise_1_experiments_path, 1, 'Iteration')
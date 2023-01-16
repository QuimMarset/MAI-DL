#!/usr/bin/env python
import tensorflow as tf
from keras.optimizer_v2.adam import Adam
import read_inputs
import numpy as np
import time



def labels_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[range(num_labels), labels] = 1
    return one_hot


def create_weight_variable(shape, stddev=0.1):
    initial = tf.random.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def create_bias_variable(shape, initial_value=0.1):
    initial = tf.constant(initial_value, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def compute_prediction_logits(batch_x, Ws, bs, keep_prob, test=False):
    h_conv1 = tf.nn.relu(conv2d(batch_x, Ws[0]) + bs[0])
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, Ws[1]) + bs[1])
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, tf.reduce_prod(h_pool2.shape[1:])])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, Ws[2]) + bs[2])

    if test is False:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

    logits = tf.matmul(h_fc1, Ws[3]) + bs[3]
    return logits


def compute_loss(labels, logits):
    cross_entropy_local = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    cross_entropy = tf.reduce_mean(cross_entropy_local)
    return cross_entropy


def update_weights(batch_x, batch_y, Ws, bs, keep_prob, optimizer):
    with tf.GradientTape() as tape:
        logits = compute_prediction_logits(batch_x, Ws, bs, keep_prob)
        loss = compute_loss(batch_y, logits)

    trainable_variables = [*Ws, *bs]
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def compute_accuracy(labels, logits):
    correct_predictions = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy



if __name__ == '__main__':

    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print(f'Num GPUs Available: {num_gpus}')

    data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
    data = data_input[0]
    side = data_input[1]
    channels = data_input[2]
    num_classes = data_input[3]

    train_x, train_y = data[0]
    test_x, test_y = data[2]
    print(train_x.shape, test_x.shape)
    print(train_y.shape, test_y.shape)

    train_y = labels_to_one_hot(train_y, num_classes)
    test_y = labels_to_one_hot(test_y, num_classes)

    # Reshape images to a 4d tensor: batch_size, width, heigh, num_channels
    # Initially, the image is flattened to a 1d vector
    train_x = tf.reshape(train_x, [-1, side, side, channels])
    test_x = tf.reshape(test_x, [-1, side, side, channels])

    # First convolutional layer: 32 features per each 5x5 patch
    W_conv1 = create_weight_variable([5, 5, 1, 32])
    b_conv1 = create_bias_variable([32])

    # Second convolutional layer: 64 features for each 5x5 patch
    W_conv2 = create_weight_variable([5, 5, 32, 64])
    b_conv2 = create_bias_variable([64])

    # First fully connected layer: Processes the 64 7x7 images with 1024 neurons
    W_fc1 = create_weight_variable([7 * 7 * 64, 1024])
    b_fc1 = create_bias_variable([1024])

    # Classification layer: Process the 1024 input neurons to compute the prediction
    W_fc2 = create_weight_variable([1024, 10])
    b_fc2 = create_bias_variable([10])

    Ws = [W_conv1, W_conv2, W_fc1, W_fc2]
    bs = [b_conv1, b_conv2, b_fc1, b_fc2]

    optimizer = Adam(1e-4)

    print("Training")

    start_time = time.time()
    batch_size = 50
    keep_prob = 0.5

    for i in range(1000):
        start = batch_size * i
        end = start + batch_size

        for j in range(num_gpus):
            with tf.device(f'/gpu:{j}'):
                batch_x = train_x[start : end]
                batch_y = train_y[start : end]

        if i % 10 == 0:
            logits = compute_prediction_logits(batch_x, Ws, bs, keep_prob, test=True)
            train_accuracy = compute_accuracy(batch_y, logits)
            print(f'Step {i}: Train accuracy: {train_accuracy:.2f}, Batch: [{start}, {end}]')

        update_weights(batch_x, batch_y, Ws, bs, keep_prob, optimizer)

    print(f'Training time: {time.time() - start_time:.3f} seconds')

    print("Testing")

    logits = compute_prediction_logits(test_x, Ws, bs, keep_prob, test=True)
    test_accuracy = compute_accuracy(test_y, logits)
    print(f'Test accuracy: {test_accuracy:.3f}')
import tensorflow as tf
from keras.optimizer_v2.gradient_descent import SGD
from keras.losses import CategoricalCrossentropy
import numpy as np
import read_inputs



def labels_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[range(num_labels), labels] = 1
    return one_hot


def compute_predictions(batch_x, W, b):
    return tf.nn.softmax(tf.matmul(batch_x, W) + b)


def compute_loss(batch_y, predictions):
    batch_y = tf.convert_to_tensor(batch_y, tf.float32)
    loss = tf.reduce_mean(-tf.reduce_sum(batch_y * tf.math.log(predictions), axis=1))
    return loss


def update_weights(batch_x, batch_y, W, b, optimizer):    
    with tf.GradientTape() as tape:
        predictions = compute_predictions(batch_x, W, b)
        loss = compute_loss(batch_y, predictions)

    trainable_variables = [W, b]
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))



if __name__ == '__main__':

    data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
    data = data_input[0]
    num_classes = data_input[3]

    train_x, train_y = data[0]
    test_x, test_y = data[2]
    print(train_x.shape, test_x.shape)
    print(train_y.shape, test_y.shape)

    train_y_one_hot = labels_to_one_hot(train_y, num_classes)

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    optimizer = SGD(0.5)

    print("Training")

    batch_size = 100

    for i in range(500):
        start = batch_size * i
        end = start + batch_size

        batch_x = train_x[start : end]
        batch_y = train_y_one_hot[start : end]
        update_weights(batch_x, batch_y, W, b, optimizer)

    print("Testing")

    test_predictions = compute_predictions(test_x, W, b)
    test_predictions = tf.argmax(test_predictions, axis=1)

    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(test_y, test_predictions), tf.float32))
    print(f'Test accuracy: {test_accuracy.numpy():.4f}')
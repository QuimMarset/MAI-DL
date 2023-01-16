import tensorflow as tf
import numpy as np
from keras.optimizer_v2.gradient_descent import SGD



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
    print(f'W: {W.numpy()}, b: {b.numpy()}, Loss: {loss.numpy()}')



if __name__ == '__main__':

    # Model parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)

    x_train = np.array([1, 2, 3, 4])
    y_train = np.array([0, -1, -2, -3])

    optimizer = SGD(0.01)

    initial_loss = compute_loss(x_train, y_train, W, b)
    print(f'Initial values: W: {W}, b: {b}, Initial loss: {initial_loss}')

    for i in range(1000):
        update_weights(x_train, y_train, W, b, optimizer)
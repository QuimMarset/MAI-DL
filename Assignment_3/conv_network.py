import tensorflow as tf
import time
import numpy as np
from tensorflow import keras



class ConvNetwork:

    def __init__(self, image_shape, kernel_size, convs_filters, denses_units, keep_prob, num_classes, l2_coef, seed):
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)
        self.keep_prob = keep_prob
        self.l2_coef = l2_coef
        self.create_trainable_variables(image_shape, kernel_size, convs_filters, denses_units, num_classes)
        self.rng = np.random.default_rng(seed)
        self.rng_labels = np.random.default_rng(seed)
        

    def create_weight_variable(self, shape, stddev=0.1):
        initial = tf.random.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)


    def create_bias_variable(self, shape, initial_value=0.1):
        initial = tf.constant(initial_value, shape=shape)
        return tf.Variable(initial)


    def create_conv_trainable_variables(self, num_channels, kernel_size, conv_filters):
        self.conv_Ws = []
        self.conv_bs = []
        prev_channels = num_channels
        for filters in conv_filters:
            W = self.create_weight_variable([kernel_size, kernel_size, prev_channels, filters])
            b = self.create_bias_variable([filters])
            self.conv_Ws.append(W)
            self.conv_bs.append(b)
            prev_channels = filters


    def create_dense_trainable_variables(self, image_side, dense_units):
        self.dense_Ws = []
        self.dense_bs = []
        last_feature_map_side = image_side // (2 * len(self.conv_Ws))
        prev_units = self.conv_bs[-1].shape[0] * last_feature_map_side * last_feature_map_side
        for units in dense_units:
            W = self.create_weight_variable([prev_units, units])
            b = self.create_bias_variable([units])
            self.dense_Ws.append(W)
            self.dense_bs.append(b)
            prev_units = units


    def create_output_trainable_variables(self, num_classes):
        last_units = self.dense_bs[-1].shape[0]
        self.output_W = self.create_weight_variable([last_units, num_classes])
        self.output_b = self.create_bias_variable([num_classes])


    def create_batch_norm_layers(self, num_layers):
        self.batch_norm_layers = []
        for _ in range(num_layers):
            self.batch_norm_layers.append()


    def create_trainable_variables(self, image_shape, kernel_size, conv_filters, dense_units, num_classes):
        self.create_conv_trainable_variables(image_shape[-1], kernel_size, conv_filters)
        self.create_dense_trainable_variables(image_shape[0], dense_units)
        self.create_output_trainable_variables(num_classes)
        self.trainable_variables = [*self.conv_Ws, *self.conv_bs, *self.dense_Ws, *self.dense_bs, 
            self.output_W, self.output_b]


    def conv2d(self, x, W, b):
        conv_output = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv_output + b)


    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    
    def dense(self, x, W, b, use_relu=True):
        dense_output = tf.matmul(x, W) + b
        if use_relu:
            return tf.nn.relu(dense_output)
        return dense_output

    
    def flatten(self, x):
        flattened_dims = tf.reduce_prod(x.shape[1:])
        return tf.reshape(x, [-1, flattened_dims])


    def compute_logits(self, batch_x, test=False):
        x = batch_x

        for W, b in zip(self.conv_Ws, self.conv_bs):
            x = self.conv2d(x, W, b)
            x = self.max_pool_2x2(x)

        x = self.flatten(x)

        for W, b in zip(self.dense_Ws, self.dense_bs):
            x = self.dense(x, W, b)
            if self.keep_prob > 0 and test is False:
                x = tf.nn.dropout(x, self.keep_prob)

        x = self.dense(x, self.output_W, self.output_b, use_relu=False)
        return x


    def compute_loss(self, labels, logits):
        cross_entropy_local = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        cross_entropy = tf.reduce_mean(cross_entropy_local)
        return cross_entropy

    
    def compute_l2_regularization(self):
        l2_loss = 0
        for weight in [*self.conv_Ws, *self.dense_Ws, self.output_W]:
            l2_loss += self.l2_coef * tf.nn.l2_loss(weight)
        return l2_loss


    def update_weights(self, batch_x, batch_y, optimizer, gradient_clip):
        with tf.GradientTape() as tape:
            logits = self.compute_logits(batch_x)
            loss = self.compute_loss(batch_y, logits)
            loss += self.compute_l2_regularization()

        gradients = tape.gradient(loss, self.trainable_variables)

        if gradient_clip > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
        
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    def compute_accuracy(self, labels, logits):
        correct_predictions = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy


    def get_num_batches(self, train_x, batch_size):
        num_batches = train_x.shape[0] // batch_size
        if train_x.shape[0] % batch_size != 0:
            num_batches += 1
        return num_batches


    def shuffle_train(self, train_x, train_y):
        self.rng.shuffle(train_x)
        self.rng_labels.shuffle(train_y)


    def train(self, train_x, train_y, batch_size, optimizer, gradient_clip, num_epochs):
        start_time = time.time()
        num_batches = self.get_num_batches(train_x, batch_size)
        training_accuracies = []

        for epoch in range(num_epochs):

            self.shuffle_train(train_x, train_y)

            for i in range(num_batches):
                start = batch_size * i
                end = start + batch_size
                batch_x = train_x[start : end]
                batch_y = train_y[start : end]

                #if i % 10 == 0:
                #    logits = self.compute_logits(batch_x, test=True)
                #    train_accuracy = self.compute_accuracy(batch_y, logits)
                #    training_accuracies.append(float(train_accuracy))
                #    print(f'Batch {i}: Train accuracy: {train_accuracy:.2f}, Batch: [{start}, {end}]')

                self.update_weights(batch_x, batch_y, optimizer, gradient_clip)

        training_time = time.time() - start_time
        print(f'Training time: {training_time:.3f} seconds')
        return training_time, training_accuracies


    def test(self, test_x, test_y, batch_size):
        num_batches = self.get_num_batches(test_x, batch_size)

        logits = [] 
        for i in range(num_batches):
            start = batch_size * i
            end = start + batch_size
            batch = test_x[start : end]
            logits_i = self.compute_logits(batch, test=True)
            logits.extend(logits_i.numpy().tolist())

        test_accuracy = self.compute_accuracy(test_y, logits)
        print(f'Test accuracy: {test_accuracy:.4f}')
        return float(test_accuracy)
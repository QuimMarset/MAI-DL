import tensorflow as tf
from tensorflow import keras




class ModelTesting(keras.Model):

    def __init__(self):
        super().__init__()

        initializer = kernel_initializer=keras.initializers.HeUniform(seed=1412)

        self.conv_1 = keras.layers.Conv2D(16, 3, padding='same', kernel_initializer=initializer)
        self.conv_2 = keras.layers.Conv2D(32, 3, padding='same', kernel_initializer=initializer)
        self.conv_3 = keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer)
        self.conv_4 = keras.layers.Conv2D(96, 3, padding='same', kernel_initializer=initializer)
        self.dense_1 = keras.layers.Dense(16, kernel_initializer=initializer)
        self.dense_2 = keras.layers.Dense(32, kernel_initializer=initializer)
        self.dense_3 = keras.layers.Dense(64, kernel_initializer=initializer)
        self.pool_1 = keras.layers.MaxPool2D()
        self.pool_2 = keras.layers.MaxPool2D()
        self.pool_3 = keras.layers.MaxPool2D()
        self.pool_4 = keras.layers.MaxPool2D()

        self.dense_4 = keras.layers.Dense(29, activation='softmax', kernel_initializer=keras.initializers.GlorotNormal(seed=1412))


    def train_step(self, batch_data):
        images, target = batch_data
        #target = tf.one_hot(target, 29)

        with tf.GradientTape() as tape:
            pred = self(images, training=True)
            #loss = self.calculate_loss(target, pred)
            loss = self.compiled_loss(target, pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(target, pred)
        #self.loss_metric.update_state(loss)
        #one_hot_targets = tf.one_hot(target, 29)
        #self.accuracy_metric.update_state(one_hot_targets, pred)

        return {m.name: m.result() for m in self.metrics}
        #return {'loss' : self.loss_metric.result(), 'accuracy' : self.accuracy_metric.result()}

    
    def test_step(self, batch_data):
        input, target = batch_data
        #target = tf.one_hot(target, 29)

        pred = self(input, training=False)
        loss = self.compiled_loss(target, pred, regularization_losses=self.losses)
        #loss = self.calculate_loss(target, pred)
        
        self.compiled_metrics.update_state(target, pred)
        #self.loss_metric.update_state(loss)
        #one_hot_targets = tf.one_hot(target, 29)
        #self.accuracy_metric.update_state(one_hot_targets, pred)

        return {m.name: m.result() for m in self.metrics}
        #return {'loss' : self.loss_metric.result(), 'accuracy' : self.accuracy_metric.result()}


    def call(self, images):
        x = self.conv_1(images)
        x = keras.layers.Activation('relu')(x)
        x = self.pool_1(x)
        
        x = self.conv_2(x)
        x = keras.layers.Activation('relu')(x)
        x = self.pool_2(x)
        
        x = self.conv_3(x)
        x = keras.layers.Activation('relu')(x)
        x = self.pool_3(x)

        x = self.conv_4(x)
        x = keras.layers.Activation('relu')(x)
        x = self.pool_4(x)
        
        x = keras.layers.Flatten()(x)
        
        x = self.dense_1(x)
        x = keras.layers.Activation('relu')(x)
        x = self.dense_2(x)
        x = keras.layers.Activation('relu')(x)
        x = self.dense_3(x)
        x = keras.layers.Activation('relu')(x)

        x = self.dense_4(x)

        return x
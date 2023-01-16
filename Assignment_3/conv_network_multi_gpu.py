import time
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import L2
from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.callbacks import Callback




class TrainingTimeCallback(Callback):

    def __init__(self):
        super().__init__()
        self.times_per_epoch = []
        self.start_time_all = 0
        self.start_time_epoch = 0


    def on_train_begin(self, logs=None):
        self.start_time_all = time.time()


    def on_train_end(self, logs=None):
        self.total_time_all = time.time() - self.start_time_all
        print(f'Total training time: {self.total_time_all:.2f}')
        print(f'Total training time (only counting epoch time): {sum(self.total_time_epoch):.2f}')


    def on_epoch_begin(self, epoch, logs={}):
        self.start_time_epoch = time.time()


    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time.time() - self.start_time_epoch
        self.times_per_epoch.append(epoch_time)



class ConvNetworkMultiGPU:


    def __init__(self, image_shape, kernel_size, convs_filters, denses_units, keep_prob, num_classes, l2_coef, seed):
        tf.random.set_seed(seed)
        
        self.keep_prob = keep_prob
        self.l2_coef = l2_coef
        self.strategy = tf.distribute.MirroredStrategy()

        #with self.strategy.scope():
        self.create_model(image_shape, kernel_size, convs_filters, denses_units, keep_prob, num_classes)


    def create_model(self, image_shape, kernel_size, conv_filters, dense_units, keep_prob, num_classes):
        input = Input(image_shape)
        x = input
        
        for filters in conv_filters:
            x = Conv2D(filters, kernel_size, activation='relu', kernel_regularizer=L2(self.l2_coef))(x)
            x = MaxPool2D()(x)

        x = Flatten()(x)

        for units in dense_units:
            x = Dense(units, activation='relu', kernel_regularizer=L2(self.l2_coef))(x)
            x = Dropout(keep_prob)(x)

        x = Dense(num_classes, activation='softmax', kernel_regularizer=L2(self.l2_coef))(x)
        self.model = Model(input, x)

    
    def compile(self, optimizer):
        #with self.strategy.scope():
        self.model.compile(optimizer, run_eagerly=False, loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])


    def train(self, train_x, train_y, batch_size, epochs):
        #start_time = time.time()
        
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2, 
            callbacks=[TrainingTimeCallback()])
        
        #training_time = time.time() - start_time
        #print(f'Training time: {training_time} seconds')


    def test(self, test_x, test_y, batch_size):
        _, test_accuracy = self.model.evaluate(test_x, test_y, batch_size, verbose=0)
        print(f'Test accuracy: {test_accuracy}')
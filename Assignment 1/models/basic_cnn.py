import tensorflow as tf
from tensorflow import keras
import os
from utils.file_io_utils import write_json_string



class CNNModel(tf.keras.Model):
    
    def __init__(self, kernel_size, filters_list, dense_units_list, use_batch_norm, dropout_percentage, use_max_pooling, num_classes):
        super().__init__()
        self.conv_layers = []
        self.dense_layers = []
        self.num_classes = num_classes
        self.create_convolutional_blocks(kernel_size, filters_list, use_batch_norm, use_max_pooling)
        self.flatten_layer = keras.layers.Flatten()
        self.dropout_layer = keras.layers.Dropout(dropout_percentage)
        self.create_classification_block(dense_units_list, use_batch_norm)
        

    def create_convolutional_blocks(self, kernel_size, filters_list, use_batch_norm, use_max_pooling):
        for filters in filters_list:
            if use_batch_norm:
                conv_block = keras.Sequential([
                    keras.layers.Conv2D(kernel_size, filters),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation('relu')
                ])
            else:
                conv_block = keras.layers.Conv2D(kernel_size, filters, activation='relu')
            if use_max_pooling:
                pooling_layer = keras.layers.MaxPooling2D()  
            else:
                pooling_layer = keras.layers.AvgPooling2D()
            self.conv_layers.append(conv_block)
            self.conv_layers.append(pooling_layer)


    def create_classification_block(self, dense_units_list, use_batch_norm):
        for dense_units in dense_units_list:
            if use_batch_norm:
                dense_block = keras.Sequential([
                    keras.layers.Dense(dense_units),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation('relu')
                ])
            else:
                dense_block = keras.layers.Dense(dense_units, activation='relu')
            self.dense_layers.append(dense_block)
        self.dense_layers.append(keras.layers.Dense(self.num_classes, activation='softmax'))


    def call(self, images):
        x = images

        for conv_block in self.conv_layers:
            x = conv_block(x)

        x = self.flatten_layer(x)

        for dense_block in self.dense_layers[:-1]:
            x = dense_block(x)

        x = self.dense_layers[-1](x)
        return x

    
    def save_architecture(self, save_path):
        model_json_string = self.to_json()
        write_json_string(model_json_string, os.path.join(save_path, 'model_architecture.json'))


    def load_model_weights(self, load_path):
        self.load_weights(os.path.join(load_path, 'model_weights'))


    def save_model_weights(self, save_path):
        self.save_weights(os.path.join(save_path, 'model_weights'))
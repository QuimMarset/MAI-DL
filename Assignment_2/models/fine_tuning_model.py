import numpy as np
import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout
from models.pre_trained_models import create_fine_tuning_base
from utils.file_io_utils import load_json_to_string, write_json_string
from utils.path_utils import join_path



class FineTuningModel:


    def __init__(self, pre_trained_name, image_shape, num_to_freeze, dense_list, 
        batch_norm, dropout, activation, num_classes, seed):
        
        self.seed = seed
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)
        self.create_model(pre_trained_name, image_shape, num_to_freeze, dense_list, 
            batch_norm, dropout, activation, num_classes)


    def create_model(self, pre_trained_name, image_shape, num_to_freeze, dense_list, 
        batch_norm, dropout, activation, num_classes):

        fine_tuning_base = create_fine_tuning_base(pre_trained_name, image_shape, num_to_freeze)
        output_shape = fine_tuning_base.output.shape[1:]
        classification_head = self.__create_classification_head(output_shape, dense_list, batch_norm, 
            dropout, activation, num_classes)

        input = fine_tuning_base.input
        base_output = fine_tuning_base.output
        classif_output = classification_head(base_output)
        self.model = keras.Model(input, classif_output)

    
    def compile(self, optimizer, label_smoothing):
        loss_function = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        self.model.compile(optimizer, run_eagerly=True, loss=loss_function, metrics=["accuracy"])


    def fit(self, train_gen, epochs, val_gen, patience):
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, 
            restore_best_weights=True, min_delta=1e-5)

        history = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen, 
            workers=6, callbacks=[early_stopping], shuffle=False)

        return history

    
    def predict(self, test_gen):
        predictions = self.model.predict(test_gen)
        return np.argmax(predictions, axis=-1)


    @classmethod
    def load_model(cls, load_path):
        instance = cls.__new__(cls)
        instance.__load_architecture(load_path)
        instance.__load_weights(load_path)
        return instance


    def save_model(self, save_path):
        self.__save_summary(save_path)
        self.__save_architecture(save_path)
        self.__save_weights(save_path)


    def __create_classification_head(self, input_shape, dense_list, batch_norm, dropout, activation, num_classes):
        input = keras.Input(input_shape)
        x = keras.layers.Flatten()(input)
        weight_initializer = keras.initializers.HeUniform(seed=self.seed)
        
        for dense_units in dense_list:
            if batch_norm:
                dense_block = self.__create_batch_norm_block(dense_units, activation, weight_initializer)
            else:
                dense_block = keras.layers.Dense(dense_units, activation=activation, kernel_initializer=weight_initializer)

            x = dense_block(x)

            if dropout > 0:
                x = keras.layers.Dropout(dropout, seed=self.seed)(x)

        initializer = keras.initializers.GlorotUniform(seed=self.seed)
        classif_output = keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer)(x)
        return keras.Model(input, classif_output)
    

    def __create_batch_norm_block(self, units, activation, weight_initializer):
        return keras.Sequential([
            keras.layers.Dense(units, kernel_initializer=weight_initializer, use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation)
        ])


    def __save_summary(self, save_path):
        file_path = join_path(save_path, 'model_summary.txt')
        with open(file_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary(expand_nested=True)


    def __load_architecture(self, load_path):
        file_path = join_path(load_path, 'model_architecture.json')
        architecture = load_json_to_string(file_path)
        self.model =  keras.models.model_from_json(architecture)
        
    
    def __save_architecture(self, save_path):
        model_json_string = self.model.to_json()
        file_path = join_path(save_path, 'model_architecture.json')
        write_json_string(model_json_string, file_path)


    def __load_weights(self, load_path):
        file_path = join_path(load_path, 'model_architecture.json')
        self.model.load_weights(file_path).expect_partial()


    def __save_weights(self, save_path):
        file_path = join_path(save_path, 'model_architecture.json')
        self.model.save_weights(file_path)
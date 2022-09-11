from tensorflow import keras
import os
from contextlib import redirect_stdout
from utils.file_io_utils import write_json_string
from utils.model_utils import *
from utils.paths_utils import join_path



class CNNModelWrapper:
    
    def __init__(self, input_shape, kernel_sizes, filters_list, padding_type, dense_units_list, use_batch_norm, dropout_percentage, 
        use_max_pooling, num_classes, seed=1412):
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.seed = seed
        self.create_model(kernel_sizes, filters_list, padding_type, dense_units_list, use_batch_norm, dropout_percentage, use_max_pooling)


    def create_model(self, kernel_sizes, filters_list, padding_type, dense_units_list, use_batch_norm, dropout_percentage, use_max_pooling):
        input = keras.Input(self.input_shape)
        flattened_output = create_convolutional_blocks(input, kernel_sizes, filters_list, padding_type, use_batch_norm, use_max_pooling, self.seed)
        output = create_classification_head(flattened_output, dense_units_list, use_batch_norm, dropout_percentage, self.num_classes, self.seed)
        self.model = keras.Model(input, output)


    def save_summary(self, save_path):
        file_path = join_path(save_path, 'model_summary.txt')
        with open(file_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary(expand_nested=True)

    
    def save_architecture(self, save_path):
        model_json_string = self.model.to_json()
        write_json_string(model_json_string, os.path.join(save_path, 'model_architecture.json'))


    def load_model_weights(self, load_path):
        self.model.load_weights(os.path.join(load_path, 'model_weights'))


    def save_model_weights(self, save_path):
        self.model.save_weights(os.path.join(save_path, 'model_weights'))
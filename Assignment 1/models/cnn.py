from tensorflow import keras
from models.basic_model import BasicModelWrapper
from utils.model_utils import *



class CNNWrapper(BasicModelWrapper):
    
    def __init__(self, input_shape, kernel_sizes, filters_list, padding_type, dense_units_list, 
        use_batch_norm, dropout_percentage, use_max_pooling, activation, num_classes, global_seed, operational_seed):

        super().__init__(input_shape, num_classes, activation, global_seed, operational_seed)
        self.create_model(kernel_sizes, filters_list, padding_type, dense_units_list, 
            use_batch_norm, dropout_percentage, use_max_pooling, activation)


    def create_model(self, kernel_sizes, filters_list, padding_type, dense_units_list, use_batch_norm, 
        dropout_percentage, use_max_pooling, activation):
        
        input = keras.Input(self.input_shape)

        flattened_output = create_convolutional_blocks(input, kernel_sizes, filters_list, padding_type, use_batch_norm, 
            use_max_pooling, activation, self.weight_initializer)

        output = create_classification_head(flattened_output, dense_units_list, use_batch_norm, dropout_percentage, 
            activation, self.num_classes, self.weight_initializer, self.operational_seed)
            
        self.model = keras.Model(input, output)
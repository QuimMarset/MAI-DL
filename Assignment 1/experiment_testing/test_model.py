import sys
sys.path.append('./')

from models.cnn import CNNWrapper
from models.cnn_residuals import CNNResidualsWrapper



if __name__ == '__main__':


    input_shape = (256, 256, 3)
    kernel_sizes = [3, 3, 3]
    filters_list = [16, 32, 64]
    padding_type = 'same'
    dense_units_list = [32]
    use_batch_norm = False
    dropout_percentage = 0
    use_max_pooling = True
    activation = 'relu'
    num_classes = 29


    model_wrapper = CNNResidualsWrapper(input_shape, kernel_sizes, filters_list, 
        dense_units_list, use_batch_norm, dropout_percentage, use_max_pooling, activation, num_classes)
    print(model_wrapper.model.summary(expand_nested=True))
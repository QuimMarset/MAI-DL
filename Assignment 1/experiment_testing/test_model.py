from models.cnn import CNNWrapper



if __name__ == '__main__':


    input_shape = (256, 256, 3)
    kernel_sizes = [3]
    filters_list = [16, 32, 64, 128]
    padding_type = 'same'
    dense_units_list = [128, 256]
    use_batch_norm = False
    dropout_percentage = 0
    use_max_pooling = True
    activation = 'relu'
    num_classes = 29


    model_wrapper = CNNWrapper(input_shape, kernel_sizes, filters_list, padding_type, 
        dense_units_list, use_batch_norm, dropout_percentage, use_max_pooling, activation, num_classes)
    print(model_wrapper.model.summary())
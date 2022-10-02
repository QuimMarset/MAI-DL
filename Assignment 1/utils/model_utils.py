from tensorflow import keras


def create_kernel_initializer(activation, seed):
    if activation == 'relu':
        return keras.initializers.HeUniform(seed=seed)
    else:
        return keras.initializers.GlorotUniform(seed=seed)


def create_conv_batch_norm_block(kernel_size, filters, padding_type, activation, seed):
    return keras.Sequential([
        keras.layers.Conv2D(filters, kernel_size, padding=padding_type, 
            kernel_initializer=create_kernel_initializer(activation, seed), use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation(activation)
    ])


def create_conv_residual_block(input_shape, kernel_size, filters, activation, seed):
    input = keras.Input(input_shape)
    matched_input = keras.layers.Conv2D(filters, 1, kernel_initializer=create_kernel_initializer(activation, seed))(input)

    x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, 
        kernel_initializer=create_kernel_initializer(activation, seed))(input)
    x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, 
        kernel_initializer=create_kernel_initializer(activation, seed))(x)
    
    x = x + matched_input
    output = keras.layers.Activation(activation)(x)
    return keras.Model(input, output)


def create_conv_batch_norm_residual_block(input_shape, kernel_size, filters, activation, seed):
    input = keras.Input(input_shape)
    matched_input = keras.layers.Conv2D(filters, 1, kernel_initializer=create_kernel_initializer(activation, seed))(input)
    matched_input = keras.layers.BatchNormalization()(matched_input)

    x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, 
        kernel_initializer=create_kernel_initializer(activation, seed), use_bias=False)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, 
        kernel_initializer=create_kernel_initializer(activation, seed), use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = x + matched_input
    output = keras.layers.Activation(activation)(x)
    return keras.Model(input, output)


def create_pooling_layer(use_max_pooling):
    if use_max_pooling:
        return keras.layers.MaxPooling2D()  
    else:
        return keras.layers.AveragePooling2D()
    

def create_dense_batch_norm_block(units, activation, seed):
    return keras.Sequential([
        keras.layers.Dense(units, kernel_initializer=create_kernel_initializer(activation, seed), use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation(activation)
    ])


def create_convolutional_blocks(input, kernel_sizes, filters_list, padding_type, use_batch_norm, use_max_pooling, activation, seed):
    x = input

    for filters, kernel_size in zip(filters_list, kernel_sizes):
        if use_batch_norm:
            conv_block = create_conv_batch_norm_block(kernel_size, filters, padding_type, activation, seed)
        else:
            conv_block = keras.layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=create_kernel_initializer(activation, seed))

        pooling_layer = create_pooling_layer(use_max_pooling)  
        
        x = conv_block(x)
        x = pooling_layer(x)

    flattened_output = keras.layers.Flatten()(x)
    return flattened_output


def create_classification_head(flattened_input, dense_units_list, use_batch_norm, dropout_percentage, activation, num_classes, seed):
    x = flattened_input

    for dense_units in dense_units_list:
        if use_batch_norm:
            dense_block = create_dense_batch_norm_block(dense_units, activation, seed)
        else:
            dense_block = keras.layers.Dense(dense_units, activation=activation, kernel_initializer=create_kernel_initializer(activation, seed))

        x = dense_block(x)

        if dropout_percentage > 0:
            x = keras.layers.Dropout(dropout_percentage, seed=seed)(x)

    output = keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=keras.initializers.GlorotUniform(seed=seed))(x)
    return output


def create_conv_residual_blocks(input, kernel_sizes, filters_list, use_batch_norm, use_max_pooling, activation, seed):
    x = input

    for filters, kernel_size in zip(filters_list, kernel_sizes):
        input_shape = x.shape[1:]

        if use_batch_norm:
            conv_block = create_conv_batch_norm_residual_block(input_shape, kernel_size, filters, activation, seed)
        else:
            conv_block = create_conv_residual_block(input_shape, kernel_size, filters, activation, seed)

        pooling_layer = create_pooling_layer(use_max_pooling)  
        
        x = conv_block(x)
        x = pooling_layer(x)

    flattened_output = keras.layers.Flatten()(x)
    return flattened_output
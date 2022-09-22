from tensorflow import keras


def create_kernel_initializer(activation, seed):
    if activation == 'relu':
        return keras.initializers.HeUniform(seed=seed)
    else:
        return keras.initializers.GlorotUniform(seed=seed)


def create_conv_batch_norm_block(kernel_size, filters, padding_type, activation, kernel_initializer):
    return keras.Sequential([
        keras.layers.Conv2D(filters, kernel_size, padding=padding_type, kernel_initializer=kernel_initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Activation(activation)
    ])


def create_conv_residual_block(input_shape, kernel_size, filters, activation, kernel_initializer):
    input = keras.Input(input_shape)
    matched_input = keras.layers.Conv2D(filters, 1, kernel_initializer=kernel_initializer)(input)

    x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, kernel_initializer=kernel_initializer)(input)
    x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, kernel_initializer=kernel_initializer)(x)
    
    x = x + matched_input
    output = keras.layers.Activation(activation)
    return keras.Model(input, output)


def create_conv_batch_norm_residual_block(input_shape, kernel_size, filters, activation, kernel_initializer):
    input = keras.Input(input_shape)
    matched_input = keras.layers.Conv2D(filters, 1, kernel_initializer=kernel_initializer)(input)
    matched_input = keras.layers.BatchNormalization()(matched_input)

    x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, kernel_initializer=kernel_initializer)
    x = keras.layers.BatchNormalization()
    x = keras.layers.Activation(activation)
    x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, kernel_initializer=kernel_initializer)
    x = keras.layers.BatchNormalization()
    
    x = x + matched_input
    output = keras.layers.Activation(activation)
    return keras.Model(input, output)


def create_pooling_layer(use_max_pooling):
    if use_max_pooling:
        return keras.layers.MaxPooling2D()  
    else:
        return keras.layers.AveragePooling2D()
    

def create_dense_batch_norm_block(units, activation, kernel_initializer):
    return keras.Sequential([
        keras.layers.Dense(units, kernel_initializer=kernel_initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Activation(activation)
    ])


def create_convolutional_blocks(input, kernel_sizes, filters_list, padding_type, use_batch_norm, use_max_pooling, activation, seed):
    x = input
    kernel_initializer = create_kernel_initializer(activation, seed)

    for filters, kernel_size in zip(filters_list, kernel_sizes):
        if use_batch_norm:
            conv_block = create_conv_batch_norm_block(kernel_size, filters, padding_type, activation, kernel_initializer)
        else:
            conv_block = keras.layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)

        pooling_layer = create_pooling_layer(use_max_pooling)  
        
        x = conv_block(x)
        x = pooling_layer(x)

    flattened_output = keras.layers.Flatten()(x)
    return flattened_output


def create_classification_head(flattened_input, dense_units_list, use_batch_norm, dropout_percentage, activation, num_classes, seed):
    x = flattened_input
    kernel_initializer = create_kernel_initializer(activation, seed)

    for dense_units in dense_units_list:
        if use_batch_norm:
            dense_block = create_dense_batch_norm_block(dense_units, activation, kernel_initializer)
        else:
            dense_block = keras.layers.Dense(dense_units, activation=activation, kernel_initializer=keras.initializers.HeUniform(seed=seed))

        x = dense_block(x)

        if dropout_percentage > 0:
            x = keras.layers.Dropout(dropout_percentage, seed=seed)(x)

    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    return output


def create_conv_residual_blocks(input, kernel_sizes, filters_list, use_batch_norm, use_max_pooling, activation, seed):
    x = input
    kernel_initializer = create_kernel_initializer(activation, seed)

    for filters, kernel_size in zip(filters_list, kernel_sizes):
        if use_batch_norm:
            conv_block = create_conv_batch_norm_residual_block(kernel_size, filters, activation, kernel_initializer)
        else:
            conv_block = create_conv_batch_norm_block(kernel_size, filters, activation, kernel_initializer)

        pooling_layer = create_pooling_layer(use_max_pooling)  
        
        x = conv_block(x)
        x = pooling_layer(x)

    flattened_output = keras.layers.Flatten()(x)
    return flattened_output
import tensorflow as tf
from tensorflow import keras



def create_convolutional_blocks(input, kernel_sizes, filters_list, padding_type, use_batch_norm, use_max_pooling, seed):
        x = input

        if len(kernel_sizes) == 1:
            kernel_sizes = kernel_sizes * len(filters_list)

        for filters, kernel_size in zip(filters_list, kernel_sizes):
            if use_batch_norm:
                conv_block = keras.Sequential([
                    keras.layers.Conv2D(filters, kernel_size, kernel_initializer=keras.initializers.HeUniform(seed=seed), padding=padding_type),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation('relu')
                ])
            else:
                conv_block = keras.layers.Conv2D(filters, kernel_size, activation='relu', 
                    kernel_initializer=keras.initializers.HeUniform(seed=seed), padding=padding_type)

            if use_max_pooling:
                pooling_layer = keras.layers.MaxPooling2D()  
            else:
                pooling_layer = keras.layers.AveragePooling2D()

            x = conv_block(x)
            x = pooling_layer(x)

        flattened_output = keras.layers.Flatten()(x)
        return flattened_output


def create_classification_head(flattened_input, dense_units_list, use_batch_norm, dropout_percentage, num_classes, seed):
    x = flattened_input

    for dense_units in dense_units_list:
        if use_batch_norm:
            dense_block = keras.Sequential([
                keras.layers.Dense(dense_units, kernel_initializer=keras.initializers.HeUniform(seed=seed)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu')
            ])
        else:
            dense_block = keras.layers.Dense(dense_units, activation='relu', kernel_initializer=keras.initializers.HeUniform(seed=seed))

        x = dense_block(x)

        if dropout_percentage > 0:
            x = keras.layers.Dropout(dropout_percentage, seed=seed)(x)

    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    return output
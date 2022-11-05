import tensorflow as tf
from tensorflow import keras



if __name__ == '__main__':

    model = keras.applications.vgg16.VGG16(include_top=True)
    layer = model.layers[-2]

    weight_initializer = keras.initializers.HeUniform(seed=1234)

    for random_layer in ['fc1', 'fc2']:
        layer = model.get_layer(random_layer)
        layer.set_weights([weight_initializer(layer.kernel.shape), weight_initializer(layer.bias.shape)])
        #layer.kernel = weight_initializer(layer.kernel.shape)
        #layer.bias = weight_initializer(layer.bias.shape)

    model.summary()
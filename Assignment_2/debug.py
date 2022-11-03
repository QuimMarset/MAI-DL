import tensorflow as tf
from tensorflow import keras



if __name__ == '__main__':

    model = keras.applications.vgg16.VGG16(include_top=False)

    print()
import tensorflow as tf
from keras.applications.resnet import ResNet50
from models.pre_trained_models import *


if __name__ == '__main__':

    model = ResNet50(include_top=False, input_shape=(256, 256, 3))

    model.summary()



"""
pool1_pool
conv2_block3_out
conv3_block4_out
conv4_block4_out
conv5_block3_out

"""
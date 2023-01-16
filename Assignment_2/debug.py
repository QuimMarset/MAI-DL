import tensorflow as tf
import keras
from utils.file_io_utils import load_npy_file_to_np_array
from utils.constants import *
from utils.path_utils import join_path


if __name__ == '__main__':

    model = keras.applications.vgg16.VGG16(include_top=True)
    model.summary()
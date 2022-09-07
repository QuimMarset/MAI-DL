import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils.image_utils import *



class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, images_path, file_names, image_size, labels, num_classes):
        self.images_path = images_path
        self.batch_size = batch_size
        self.image_shape = (image_size, image_size)
        self.file_names = file_names
        self.labels = labels
        self.num_classes = num_classes

        self.num_batches = len(self.file_names) // self.batch_size
        self.on_epoch_end()


    def __len__(self):
        return self.num_batches


    def __getitem__(self, index):
        batch_index = index * self.batch_size
        images, labels = self.load_batch(batch_index)
        return images, labels


    def on_epoch_end(self):
        np.random.shuffle(self.file_names)


    def load_batch(self, batch_index):
        batch_images = np.zeros((self.batch_size, *self.image_shape, 3))
    
        batch_file_names = self.file_names[batch_index : batch_index + self.batch_size]
        batch_labels = self.labels[batch_index : batch_index + self.batch_size]
        batch_labels_one_hot = keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        for (i, file_name) in enumerate(batch_file_names):
            image = load_image(self.images_path, file_name)
            image = image_to_float(image)
            image = resize_image(image, self.image_shape)
            batch_images[i] = image

        return batch_images, batch_labels_one_hot
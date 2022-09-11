import numpy as np
from tensorflow import keras
from utils.image_utils import *



class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, images_path, file_names, image_shape, labels, num_classes, mean_std, seed=1412):
        self.images_path = images_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.file_names = file_names
        self.labels = labels
        self.num_classes = num_classes
        self.mean_std = mean_std
        self.num_batches = len(self.file_names) // self.batch_size
        self.rng_data = np.random.default_rng(seed)
        self.rng_labels = np.random.default_rng(seed)
        self.on_epoch_end()


    def __len__(self):
        return self.num_batches


    def __getitem__(self, index):
        batch_index = index * self.batch_size
        images, labels = self.load_batch(batch_index)
        return images, labels


    def on_epoch_end(self):
        self.rng_data.shuffle(self.file_names)
        self.rng_labels.shuffle(self.labels)


    def load_batch_images(self, batch_file_names):
        batch_images = np.zeros((self.batch_size, *self.image_shape))

        for (i, file_name) in enumerate(batch_file_names):
            image = load_image(self.images_path, file_name)
            # image = resize_image(image, self.image_shape[:-1])
            image = image_to_float(image)
            image = (image - self.mean_std[0]) / self.mean_std[1]
            batch_images[i] = image

        return batch_images


    def load_batch(self, batch_index):
        batch_file_names = self.file_names[batch_index : batch_index + self.batch_size]
        batch_labels = np.array(self.labels[batch_index : batch_index + self.batch_size])
        batch_images = self.load_batch_images(batch_file_names)
        return batch_images, batch_labels



class DataGeneratorAugmentation(DataGenerator):

    def load_batch_images(self, batch_file_names):
        batch_images = np.zeros((self.batch_size, *self.image_shape))

        for (i, file_name) in enumerate(batch_file_names):
            image = load_image(self.images_path, file_name)
            image = image_to_float(image)
            image = augment_image(image)
            image = (image - self.mean_std[0]) / self.mean_std[1]
            batch_images[i] = image

        return batch_images
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.image_utils import *



class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, images_path, file_names, image_shape, labels, num_classes, mean_std, seed=1412):
        self.images_path = images_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.mean_std = mean_std

        self.file_names = file_names
        self.labels = np.zeros((len(labels), num_classes))
        self.labels[range(len(labels)), labels] = 1
        
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
            image = image_to_float(image)
            image = (image - self.mean_std[0]) / self.mean_std[1]
            batch_images[i] = image

        return batch_images


    def load_batch(self, batch_index):
        batch_file_names = self.file_names[batch_index : batch_index + self.batch_size]
        batch_labels = self.labels[batch_index : batch_index + self.batch_size]
        batch_images = self.load_batch_images(batch_file_names)
        return batch_images, batch_labels



class DataGeneratorAugmentation(DataGenerator):

    def __init__(self, batch_size, images_path, file_names, image_shape, labels, num_classes, mean_std, seed=1412):
        super().__init__(batch_size, images_path, file_names, image_shape, labels, num_classes, mean_std, seed)

        def random_central_crop(batch):
            if np.random.rand() < 0.5:
                return tf.image.resize(tf.image.central_crop(batch, 0.75), image_shape[:2])
            return batch
        
        self.augmentation_model = keras.Sequential([
            keras.layers.RandomRotation(0.2, fill_mode='nearest', seed=seed),
            keras.layers.RandomFlip('horizontal_and_vertical', seed=seed),
            #keras.layers.RandomZoom(0.1, 0.1),
            keras.layers.RandomContrast(0.3, seed=seed),
            keras.layers.Lambda(random_central_crop),
            keras.layers.Lambda(lambda batch : (batch - self.mean_std[0]) / self.mean_std[1])
        ])
        self.augmentation_model.trainable = False

    def load_batch_images(self, batch_file_names):
        batch_images = np.zeros((self.batch_size, *self.image_shape))

        for (i, file_name) in enumerate(batch_file_names):
            image = load_image(self.images_path, file_name)
            image = image_to_float(image)
            batch_images[i] = image

        augmented_batch = self.augmentation_model(batch_images)
        return augmented_batch
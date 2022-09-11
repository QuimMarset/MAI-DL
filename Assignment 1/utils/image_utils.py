import os
import cv2
from skimage import img_as_ubyte, img_as_float32
import numpy as np
import tensorflow as tf



def load_image(images_path, image_name):
    image_path = os.path.join(images_path, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def image_to_float(image):
    return img_as_float32(image)


def image_to_uint8(image):
    return img_as_ubyte(image)


def resize_image(image, new_shape):
    # new_shape = (width, height)
    return cv2.resize(image, new_shape)


def square_center_crop_image(image, crop_size):
    x_center = image.shape[1] // 2
    y_center = image.shape[0] // 2

    half_crop = crop_size // 2

    x_start = max(x_center - half_crop, 0)
    y_start = max(y_center - half_crop, 0)

    x_end = min(x_center + half_crop, image.shape[1])
    y_end = min(y_center + half_crop, image.shape[0])

    return image[x_start:x_end, y_start:y_end]


def square_random_crop_image(image, crop_size):
    half_crop = crop_size // 2

    max_x = max(image.shape[1] - half_crop, 0)
    max_y = max(image.shape[0] - half_crop, 0)

    min_x = min(half_crop, image.shape[1])
    min_y = min(half_crop, image.shape[0])

    new_center_x = np.random.randint(min_x, max_x)
    new_center_y = np.random.randint(min_y, min_y)

    x_start = max(new_center_x - half_crop, 0)
    y_start = max(new_center_y - half_crop, 0)

    x_end = min(new_center_x + half_crop, image.shape[1])
    y_end = min(new_center_y + half_crop, image.shape[0])

    return image[x_start:x_end, y_start:y_end]



def augment_image(image):
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image.numpy()
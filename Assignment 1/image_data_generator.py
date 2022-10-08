import tensorflow as tf
from tensorflow import keras


def create_preprocess_function(mean_std):
    if mean_std:
        return lambda image: (image - mean_std[0]) / mean_std[1]
    return None


def create_image_data_generator(mean_std):
    return keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        preprocessing_function=create_preprocess_function(mean_std)
    )


def create_image_data_generator_augmentation(mean_std):
    return keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        preprocessing_function=create_preprocess_function(mean_std),
        rotation_range = 30,
        zoom_range = 0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest"
    )


def flow_generator(generator, dataframe, image_shape, batch_size, seed, shuffle):
    return generator.flow_from_dataframe(
        dataframe,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode = "categorical",
        shuffle=shuffle,
        validate_filenames=False,
        seed=seed
    )


def create_train_generator(dataframe, image_shape, batch_size, augmentation, mean_std, seed):
    if augmentation:
        generator = create_image_data_generator_augmentation(mean_std)
    else:
        generator = create_image_data_generator(mean_std)

    return flow_generator(generator, dataframe, image_shape, batch_size, seed, shuffle=True)


def create_val_generator(dataframe, image_shape, batch_size, mean_std):
    generator = create_image_data_generator(mean_std)
    return flow_generator(generator, dataframe, image_shape, batch_size, seed=None, shuffle=False)
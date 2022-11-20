from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.resnet import preprocess_input as resnet50_preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def create_preprocess_function(use_normalization, model_name):
    if use_normalization:
        if model_name == 'vgg16':
            return lambda image: vgg16_preprocess_input(image)
        elif model_name == 'vgg19':
            return lambda image: vgg19_preprocess_input(image)
        elif model_name == 'resnet_50':
            return lambda image: resnet50_preprocess_input(image)
        else:
            raise NotImplementedError(model_name)
    else:
        return None


def create_rescale_factor(use_normalization):
    if use_normalization:
        return None
    return 1.0/255.0



def create_image_data_generator(rescale_factor, preprocessing_function):
    return ImageDataGenerator(
        rescale=rescale_factor,
        preprocessing_function=preprocessing_function
    )


def create_image_data_generator_augmentation(rescale_factor, preprocessing_function):
    return ImageDataGenerator(
        rescale=rescale_factor,
        preprocessing_function=preprocessing_function,
        rotation_range = 30,
        zoom_range = 0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest"
    )


def flow_generator(generator, dataframe, image_shape, batch_size, seed, shuffle, class_mode):
    return generator.flow_from_dataframe(
        dataframe,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
        validate_filenames=False,
        seed=seed
    )


def create_train_generator(dataframe, image_shape, batch_size, augmentation, use_normalization, model_name, seed, class_mode='categorical'):
    rescale_factor = create_rescale_factor(use_normalization)
    preprocess_function = create_preprocess_function(use_normalization, model_name)

    if augmentation:
        generator = create_image_data_generator_augmentation(rescale_factor, preprocess_function)
    else:
        generator = create_image_data_generator(rescale_factor, preprocess_function)

    return flow_generator(generator, dataframe, image_shape, batch_size, seed, True, class_mode)


def create_val_generator(dataframe, image_shape, batch_size, use_normalization, model_name, class_mode='categorical'):
    rescale_factor = create_rescale_factor(use_normalization)
    preprocess_function = create_preprocess_function(use_normalization, model_name)
    generator = create_image_data_generator(rescale_factor, preprocess_function)
    return flow_generator(generator, dataframe, image_shape, batch_size, None, False, class_mode) 


def create_test_generator(dataframe, image_shape, batch_size, mean_std):
    return create_val_generator(dataframe, image_shape, batch_size, mean_std)


def create_train_gen_feat_extract(dataframe, image_shape, batch_size, use_normalization, model_name, class_mode='categorical'):
    return create_val_generator(dataframe, image_shape, batch_size, use_normalization, model_name, class_mode)
from tensorflow import keras


def __create_vgg_16(include_top, image_shape, pooling):
    return keras.applications.vgg16.VGG16(include_top=include_top, weights='imagenet', 
        input_shape=image_shape, pooling=pooling)


def __create_vgg_19(include_top, image_shape, pooling):
    return keras.applications.vgg19.VGG19(include_top=include_top, weights='imagenet', 
        input_shape=image_shape, pooling=pooling)


def __create_resnet_50(include_top, image_shape, pooling):
    return keras.applications.resnet50.ResNet50(include_top=include_top, weights='imagenet', 
        input_shape=image_shape, pooling=pooling)


def __create_pre_trained_model(model_name, include_top=False, image_shape=None, pooling=None):
    
    if model_name == 'vgg16':
        return __create_vgg_16(include_top, image_shape, pooling)
        
    elif model_name == 'vgg19':
        return __create_vgg_19(include_top, image_shape, pooling)

    elif model_name == 'resnet_50':
        return __create_resnet_50(include_top, image_shape, pooling)

    else:
        raise NotImplementedError(model_name)


def create_fine_tuning_base(model_name, image_shape, num_to_freeze, pooling=None):
    pre_trained_model = __create_pre_trained_model(model_name, include_top=False, image_shape=image_shape, pooling=pooling)
    for layer in pre_trained_model.layers[:num_to_freeze]:
        layer.trainable = False
    return pre_trained_model


def create_feature_extraction_base(model_name, feature_layers, pooling=None):
    pre_trained_model = __create_pre_trained_model(model_name, include_top=True, pooling=pooling)
    outputs = [layer.output for layer in pre_trained_model.layers if layer.name in feature_layers]
    return keras.Model(pre_trained_model.inputs, outputs)

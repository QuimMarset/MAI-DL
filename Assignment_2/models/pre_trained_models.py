from tensorflow import keras
from models.vgg_16_places import create_custom_vgg_16



def __create_vgg_16_places(include_top, image_shape):
    return create_custom_vgg_16(include_top, image_shape, is_hybrid=False)


def __create_vgg_16_hybrid(include_top, image_shape):
    return create_custom_vgg_16(include_top, image_shape, is_hybrid=True)


def __create_vgg_16(include_top, image_shape):
    return keras.applications.vgg16.VGG16(include_top=include_top, input_shape=image_shape)


def __create_vgg_19(include_top, image_shape):
    return keras.applications.vgg19.VGG19(include_top=include_top, input_shape=image_shape)


def __create_resnet_50(include_top, image_shape):
    return keras.applications.resnet50.ResNet50(include_top=include_top, input_shape=image_shape)


def __create_pre_trained_model(model_name, include_top=False, image_shape=None):
    
    if model_name == 'vgg16':
        return __create_vgg_16(include_top, image_shape)

    elif model_name == 'vgg16_places':
        return __create_vgg_16_places(include_top, image_shape)

    elif model_name == 'vgg16_hybrid':
        return __create_vgg_16_hybrid(include_top, image_shape)
        
    elif model_name == 'vgg19':
        return __create_vgg_19(include_top, image_shape)

    elif model_name == 'resnet_50':
        return __create_resnet_50(include_top, image_shape)

    else:
        raise NotImplementedError(model_name)


def freeze_layers(model, up_to_freeze):
    for layer in model.layers:
        layer.trainable = False
        if layer.name == up_to_freeze:
            return


def randomize_layers(model, random_layers, seed):
    weight_initializer = keras.initializers.HeUniform(seed=seed)
    for random_layer in random_layers:
        layer = model.get_layer(random_layer)
        layer.set_weights([weight_initializer(layer.kernel.shape), 
            weight_initializer(layer.bias.shape)])


def create_fine_tuning_base_no_FC(model_name, image_shape, up_to_freeze, random_layers, seed):
    pre_trained_model = __create_pre_trained_model(model_name, include_top=False, image_shape=image_shape)
    freeze_layers(pre_trained_model, up_to_freeze)
    randomize_layers(pre_trained_model, random_layers, seed)
    return pre_trained_model


def create_fine_tuning_base_with_FC(model_name, up_to_freeze, random_layers, seed):
    pre_trained_model = __create_pre_trained_model(model_name, include_top=True)
    pre_trained_model = keras.Model(pre_trained_model.input, pre_trained_model.layers[-2].output)
    freeze_layers(pre_trained_model, up_to_freeze)
    randomize_layers(pre_trained_model, random_layers, seed)
    return pre_trained_model


def create_feature_extraction_base(model_name, feature_layers):
    pre_trained_model = __create_pre_trained_model(model_name, include_top=True)
    outputs = [layer.output for layer in pre_trained_model.layers if layer.name in feature_layers]
    return keras.Model(pre_trained_model.inputs, outputs)

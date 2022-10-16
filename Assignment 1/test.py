from sklearn.utils import shuffle
from tensorflow import keras
import numpy as np
# import tensorflow_addons as tfa
import random
from utils.paths_utils import *
from utils.csv_utils import create_train_val_dataframes, get_classes_names
from utils.plot_utils import *
from utils.argument_parser import parse_input_arguments
from utils.file_io_utils import write_dict_to_json, load_mean_std_from_npy
from image_data_generator import create_train_generator, create_val_generator
from models import cnn, cnn_residuals
from keras.preprocessing import image



def generate_tf_random_seeds(args):
    global_seed = args.global_seed if args.seed else random.randint(0, 9999)
    operational_seed = args.operational_seed if args.seed else random.randint(0, 9999)
    return global_seed, operational_seed


def define_learning_rate(args, epoch_steps):
    if args.lr_decay:
        return keras.optimizers.schedules.ExponentialDecay(args.lr, args.decay_epochs*epoch_steps, args.decay_rate)
    else:
        return args.lr


def create_model(args, input_shape, num_classes, global_seed, operational_seed):
    use_max_pooling = args.pooling_type == 'Max'
    kernel_sizes = args.kernel_sizes * len(args.conv_filters) if len(args.kernel_sizes) == 1 else args.kernel_sizes

    if args.model == 'CNN':
        return cnn.CNNWrapper(input_shape, kernel_sizes, args.conv_filters, 
            args.padding, args.dense_units, args.batch_norm, args.dropout, use_max_pooling, 
            args.activation, num_classes, global_seed, operational_seed)
    else:
        return cnn_residuals.CNNResidualsWrapper(input_shape, kernel_sizes, args.conv_filters, 
            args.dense_units, args.batch_norm, args.dropout, use_max_pooling, args.activation, 
            num_classes, global_seed, operational_seed)


def preprocess(test_image_path):
    img = keras.utils.load_img(test_image_path, target_size=(256, 256))
    img_array = keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_norm = img_batch/max(img_batch)
    return img_norm

def test(parsed_args, weights_path, test_image_path, class_names, image_shape, num_classes, global_seed, operational_seed):

    model_wrapper = create_model(parsed_args, image_shape, num_classes, global_seed, operational_seed)
    model_wrapper.load_model_weights(weights_path)
    test_img = preprocess(test_image_path)
    prediction = model_wrapper.model.predict(test_img)
    predicted_index = np.argmax(prediction)
    print(f'Predicted class: {class_names[predicted_index]}')
    print(f'Predicted index: {predicted_index}')
    print(f'All predictions: {prediction}')



if __name__ == '__main__':
    os.environ['TF_DETERMINISTIC_OPS']  = '1'

    parsed_args = parse_input_arguments()
    
    root_path = './'
    weights_path = join_path(root_path, 'test_model')
    test_image_path = join_path(root_path, 'data_256', '212740-28575.jpg')

    classes_names = get_classes_names(join_path(root_path, 'MAMe_metadata', 'MAMe_labels.csv'))
    image_shape = (256, 256, 3)
    
    global_seed, operational_seed = generate_tf_random_seeds(parsed_args)


    test(parsed_args, weights_path, test_image_path, classes_names, image_shape, len(classes_names), global_seed, operational_seed)

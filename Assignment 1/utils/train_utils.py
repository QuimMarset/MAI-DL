from tensorflow import keras
import tensorflow_addons as tfa
from models.basic_cnn import CNNModelWrapper



def define_learning_rate(args, epoch_steps):
    if args.lr_decay:
        return keras.optimizers.schedules.ExponentialDecay(args.lr, epoch_steps, 0.95)
    else:
        return args.lr


def create_optimizer(args, epoch_steps):
    learning_rate = define_learning_rate(args, epoch_steps)

    if args.optimizer == 'SGD':
        return keras.optimizers.SGD(learning_rate)
    else:
        if args.weight_decay > 0:
            return tfa.optimizers.AdamW(weight_decay=args.weight_decay, learning_rate=learning_rate)
        else:
            return keras.optimizers.Adam(learning_rate, amsgrad=True, epsilon=0.1)


def create_model(args, input_shape, num_classes):
    use_max_pooling = args.pooling_type == 'Max'

    return CNNModelWrapper(input_shape, args.kernel_sizes, args.conv_filters, args.padding, args.dense_units, args.batch_norm,
        args.dropout, use_max_pooling, num_classes)
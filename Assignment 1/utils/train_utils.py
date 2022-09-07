from tensorflow import keras
from models.basic_cnn import CNNModel



def define_learning_rate(args):
    if args.lr_decay:
        return keras.optimizers.schedules.ExponentialDecay(args.lr, 1000, 0.95)
    else:
        return args.lr


def create_optimizer(args):
    learning_rate = define_learning_rate(args)

    if args.optimizer == 'SGD':
        return keras.optimizers.SGD(learning_rate)
    else:
        return keras.optimizers.Adam(learning_rate)


def create_model(args, num_classes):
    use_max_pooling = args.pooling == 'Max'

    return CNNModel(args.kernel_size, args.conv_filters, args.dense_units, args.batch_norm,
        args.dropout, use_max_pooling, num_classes)
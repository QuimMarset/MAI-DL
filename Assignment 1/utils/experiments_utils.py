import argparse



def parse_input_arguments():
    parser = argparse.ArgumentParser(prog='DL Assignment 1', description='Train a CNN to classify the MAMe dataset')

    # Optimizer parameters
    parser.add_argument('--optimizer', help='Optimizer to use', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', type=float, help='Learning Rate', default=1e-4)
    parser.add_argument('--lr_decay', action='store_true', help='Exponential LR decay (Decreases at each epoch)')
    parser.add_argument('--weight_decay', type=float, help='L2 weight decay regularization constant (only used with Adam)', default=0)

    # Model parameters
    parser.add_argument('--kernel_sizes', type=int, nargs='+', help='Kernel size of each convolution or one shared among all layers', default=[3])
    parser.add_argument('--conv_filters', type=int, nargs='+', help='Number of filters in each convolutional layer', default=[32, 64])
    parser.add_argument('--pooling_type', help='Pooling type to use', choices=['Max', 'Avg'])
    parser.add_argument('--padding', help='Padding to use in convolutional layers', choices=['same', 'valid'], default='same')
    parser.add_argument('--dense_units', type=int, nargs='+', help='Number of units in each dense layer', default=[128])
    parser.add_argument('--dropout', type=float, help='Dropout percentage to use after each dense layer', default=0.2)
    parser.add_argument('--batch_norm', action='store_true', help='Apply Batch Normalization after every Dense or Convolutional layer')

    # Data parameters
    parser.add_argument('--augmentation', action='store_true', help='Apply data augmentation (defined in image_utils.py)')
    parser.add_argument('--batch_size', type=int, help='Mini-batch size', default=32)

    # Other training parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=20)
    parser.add_argument('--early_stop', type=int, help='Patience value to use in validation early stopping (< 0 = no stopping)', default=-1)

    parsed_args = parser.parse_args()
    return parsed_args
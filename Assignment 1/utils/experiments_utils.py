import argparse



def create_parser():
    parser = argparse.ArgumentParser(prog='DL Assignment 1', description='Train a CNN to classify the MAMe dataset')

    # Optimizer parameters
    parser.add_argument('--optimizer', help='Optimizer to use', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', type=float, help='Learning Rate', default=1e-4)
    parser.add_argument('--lr_decay', action='store_true', help='Exponential LR decay')

    # Model parameters
    parser.add_argument('--kernel_size', type=int, help='Kernel size of all convolutions (shared)', default=3)
    parser.add_argument('--conv_filters', type=int, nargs='+', help='Number of feature maps in each convolutional layer')
    parser.add_argument('--pooling_type', help='Pooling type to use', choices=['Max', 'Avg'])
    parser.add_argument('--dense_units', type=int, nargs='+', help='Number of units in each dense layer')
    parser.add_argument('--dropout', type=float, help='Dropout percentage to use after each dense layer', default=0)
    parser.add_argument('--batch_norm', action='store_true', help='Apply Batch Normalization after every convolutional layer')

    # Data parameters
    parser.add_argument('--augmentation', action='store_true', help='Apply data augmentation (defined in .py)')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)

    # Other training parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=20)
    parser.add_argument('--early_stop', type=int, help='Patience value to use in validation early stopping (-1 = no stopping)', default=-1)
    #parser.add_argument('--weight_decay', type=float, help='L2 Regularization weight', default=0)
    parser.add_argument('--balance_classes', action='store_true', help='Balance the weights for each class in the loss computation')

    return parser
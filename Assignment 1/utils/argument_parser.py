import argparse



def parse_input_arguments():
    parser = argparse.ArgumentParser(prog='DL Assignment 1', description='Train a CNN to classify the MAMe dataset')

    # Optimizer parameters
    parser.add_argument('--optimizer', help='Optimizer to use', choices=['Adam', 'SGD'], default='Adam')
    parser.add_argument('--lr', type=float, help='Learning Rate', default=1e-3)
    parser.add_argument('--lr_decay', action='store_true', help='Use Exponential LR decay', default=False)
    parser.add_argument('--decay_epochs', type=float, help='How many epochs to exponentially decay the initial LR', default=0)
    parser.add_argument('--decay_rate', type=float, help='Which percentage to exponentially decay the initial LR at the decay epochs', default=1)
    parser.add_argument('--weight_decay', type=float, help='L2 weight decay regularization constant (only used with Adam)', default=0)

    # Model parameters
    parser.add_argument('--model', help='Model architecture to use', choices=['CNN', 'CNN-Residuals'], default='CNN')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', help='Kernel size of each convolution or one shared among all layers', default=[3])
    parser.add_argument('--conv_filters', type=int, nargs='+', help='Number of filters in each convolutional layer', default=[16, 32, 64])
    parser.add_argument('--pooling_type', help='Pooling type to use', choices=['Max', 'Avg'], default='Max')
    parser.add_argument('--padding', help='Padding to use in convolutional layers', choices=['same', 'valid'], default='same')
    parser.add_argument('--dense_units', type=int, nargs='+', help='Number of units in each dense layer', default=[128])
    parser.add_argument('--dropout', type=float, help='Dropout percentage to use after each dense layer', default=0)
    parser.add_argument('--batch_norm', action='store_true', help='Apply Batch Normalization after every Dense or Convolutional layer', default=False)
    parser.add_argument('--activation', help='Activation function to use in hidden layers', default='relu')

    # Data parameters
    parser.add_argument('--augmentation', action='store_true', help='Apply data augmentation', default=False)
    parser.add_argument('--batch_size', type=int, help='Mini-batch size', default=32)

    # Other training parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=50)
    parser.add_argument('--early_stopping', action='store_true', help='Use Early Stopping', default=False)
    parser.add_argument('--patience', type=int, help='Patience value to use in early stopping', default=10)
    parser.add_argument('--min_delta', type=float, help='Minimum change in the val loss to consider it an improvement', default=1e-4)
    parser.add_argument('--label_smoothing', type=float, help='Weight divided between the non-target labels', default=0)

    parsed_args = parser.parse_args()
    return parsed_args
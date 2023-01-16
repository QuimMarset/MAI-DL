from keras.optimizer_v2.gradient_descent import SGD
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.rmsprop import RMSProp
from keras.optimizer_v2.learning_rate_schedule import ExponentialDecay



def create_exponential_lr_decay(initial_lr, decay_steps, decay_rate):
    return ExponentialDecay(initial_lr, decay_steps, decay_rate)


def create_adam_optimizer(learning_rate, **optimizer_params):
    return Adam(learning_rate, **optimizer_params)


def create_rmsprop_optimizer(learning_rate, **optimizer_params):
    return RMSProp(learning_rate, **optimizer_params)


def create_sgd_optimizer(learning_rate, **optimizer_params):
    return SGD(learning_rate, **optimizer_params)


def create_learning_rate(config):
    learning_rate = config.learning_rate

    if config.lr_decay == 'exponential':
        return create_exponential_lr_decay(learning_rate, **config.exponential_decay_parameters)
    else:
        return learning_rate


def create_optimizer(config):
    learning_rate = create_learning_rate(config)

    if config.optimizer == 'SGD':
        return create_sgd_optimizer(learning_rate, **config.sgd_parameters)
    elif config.optimizer == 'Adam':
        return create_adam_optimizer(learning_rate, **config.adam_parameters)
    elif config.optimizer == 'RMSProp':
        return create_rmsprop_optimizer(learning_rate, **config.rmsprop_parameters)
    else:
        raise ValueError(config.optimizer)
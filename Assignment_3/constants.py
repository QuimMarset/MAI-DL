from utils.path_utils import join_path


experiments_path = 'experiments'
exercise_1_experiments_path = join_path(experiments_path, 'exercise_1')
exercise_2_experiments_path = join_path(experiments_path, 'exercise_2')
exercise_3_experiments_path = join_path(experiments_path, 'exercise_3')
exercise_4_experiments_path = join_path(experiments_path, 'exercise_4')


configs_path = 'configs'
config_1_path = join_path(configs_path, 'exercise_1_config.yaml')
config_2_path = join_path(configs_path, 'exercise_2_config.yaml')
config_3_path = join_path(configs_path, 'exercise_3_config.yaml')
config_4_path = join_path(configs_path, 'exercise_4_config.yaml')


mnist_path = join_path('MNIST_data', 'mnist.pkl.gz')
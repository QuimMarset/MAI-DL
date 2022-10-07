from contextlib import redirect_stdout
from utils.file_io_utils import write_json_string
from utils.paths_utils import join_path
from utils.model_utils import create_kernel_initializer, set_tf_global_seed



class BasicModelWrapper:
    
    def __init__(self, input_shape, num_classes, activation, global_seed, operational_seed):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.global_seed = global_seed
        self.operational_seed = operational_seed
        self.weight_initializer = create_kernel_initializer(activation, operational_seed)
        set_tf_global_seed(global_seed)


    def save_summary(self, save_path):
        file_path = join_path(save_path, 'model_summary.txt')
        with open(file_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary(expand_nested=True)

    
    def save_architecture(self, save_path):
        model_json_string = self.model.to_json()
        write_json_string(model_json_string, join_path(save_path, 'model_architecture.json'))


    def load_model_weights(self, load_path):
        self.model.load_weights(join_path(load_path, 'model_weights'))


    def save_model_weights(self, save_path):
        self.model.save_weights(join_path(save_path, 'model_weights'))
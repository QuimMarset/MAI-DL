import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout
from utils.file_io_utils import load_json_to_string, write_json_string
from utils.path_utils import join_path



class BasicModel:

    def __init__(self, seed):
        self.seed = seed
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)


    @classmethod
    def create_test_model(cls, model_path):
        instance = super().__new__(cls)
        instance.load_architecture(model_path)
        instance.load_weights(model_path)
        return instance


    def save_summary(self, save_path):
        file_path = join_path(save_path, 'model_summary.txt')
        with open(file_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary(expand_nested=True)


    def load_architecture(self, load_path):
        file_path = join_path(load_path, 'model_architecture.json')
        architecture = load_json_to_string(file_path)
        self.model =  keras.models.model_from_json(architecture)
        
    
    def save_architecture(self, save_path):
        model_json_string = self.model.to_json()
        file_path = join_path(save_path, 'model_architecture.json')
        write_json_string(model_json_string, file_path)


    def load_weights(self, load_path):
        file_path = join_path(load_path, 'model_architecture.json')
        self.model.load_weights(file_path).expect_partial()


    def save_weights(self, save_path):
        file_path = join_path(save_path, 'model_architecture.json')
        self.model.save_weights(file_path)


    def compile(self, optimizer, label_smoothing):
        loss_function = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        self.model.compile(optimizer, run_eagerly=True, loss=loss_function, metrics=["accuracy"])


    def fit(self, train_gen, epochs, val_gen, patience):
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, 
            restore_best_weights=True, min_delta=1e-5)

        history = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen, 
            workers=6, callbacks=[early_stopping], shuffle=False)

        return history

    
    def predict(self, test_gen):
        predictions = self.model.predict(test_gen)
        return np.argmax(predictions, axis=-1)
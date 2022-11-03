from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.svm import LinearSVC
from models.pre_trained_models import create_feature_extraction_base
from contextlib import redirect_stdout
from utils.file_io_utils import load_json_to_string, load_pkl_object, save_object_to_pkl, write_json_string
from utils.path_utils import join_path



class FeatureExtractionModel:

    def __init__(self, pre_trained_name, feature_layers, num_classes, seed):
        self.seed = seed
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)
        self.num_classes = num_classes
        self.feature_extractor = create_feature_extraction_base(pre_trained_name, feature_layers)
        self.classifier = LinearSVC()


    def train_classifier(self, train_gen, train_labels, val_gen):
        train_features = self.__extract_features(train_gen)
        self.classifier.fit(train_features, train_labels)
        train_predictions = self.classifier.predict(train_features)
        val_predictions = self.predict(val_gen)
        return train_predictions, val_predictions


    def predict(self, generator):
        test_features = self.__extract_features(generator)
        return self.classifier.predict(test_features)


    @classmethod
    def load_model(cls, load_path):
        instance = super().__new__(cls)
        instance.__load_architecture(load_path)
        instance.__load_weights(load_path)
        instance.classifier = load_pkl_object(join_path(load_path, 'classifier.pkl'))
        return instance

    
    def save_model(self, save_path):
        self.__save_summary(save_path)
        self.__save_architecture(save_path)
        self.__save_weights(save_path)
        save_object_to_pkl(self.classifier, join_path(save_path, 'classifier.pkl'))


    def __extract_features(self, data_gen):
        features = []
        for _ in range(len(data_gen)):
            batch_images, _ = next(data_gen)
            features.extend(self.__extract_batch_features(batch_images))
        return np.array(features)


    def __extract_batch_features(self, batch_images):
        batch_features = []
        for batch_layer_output in self.feature_extractor(batch_images):
            batch_layer_output = batch_layer_output.numpy()

            if len(batch_layer_output.shape) == 4:
                # spatial average pooling
                batch_layer_output = np.mean(batch_layer_output, axis=(1, 2))

            batch_features.append(batch_layer_output)

        batch_features = np.concatenate(batch_features, axis=-1)
        return batch_features

    
    def __standarize_features(self, features):
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0)
        standarized_features = np.divide(features - self.means, self.stds, where=self.stds != 0)
        return standarized_features


    def __save_summary(self, save_path):
        file_path = join_path(save_path, 'feature_extractor_summary.txt')
        with open(file_path, 'w') as file:
            with redirect_stdout(file):
                self.feature_extractor.summary(expand_nested=True)


    def __load_architecture(self, load_path):
        file_path = join_path(load_path, 'feature_extractor_architecture.json')
        architecture = load_json_to_string(file_path)
        self.feature_extractor =  keras.models.model_from_json(architecture)
        
    
    def __save_architecture(self, save_path):
        model_json_string = self.feature_extractor.to_json()
        file_path = join_path(save_path, 'feature_extractor_architecture.json')
        write_json_string(model_json_string, file_path)


    def __load_weights(self, load_path):
        file_path = join_path(load_path, 'feature_extractor_weights')
        self.feature_extractor.load_weights(file_path).expect_partial()


    def __save_weights(self, save_path):
        file_path = join_path(save_path, 'feature_extractor_weights')
        self.feature_extractor.save_weights(file_path)


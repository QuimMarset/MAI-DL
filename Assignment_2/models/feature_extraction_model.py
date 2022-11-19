from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
from models.pre_trained_models import create_feature_extraction_base
from contextlib import redirect_stdout
from utils.file_io_utils import *
from utils.path_utils import join_path



class FeatureExtractionModel:

    def __init__(self, pre_trained_name, feature_layers, num_classes, standarization, 
        discretization, negative_threshold, positive_threshold, seed):
        
        self.seed = seed
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)
        self.num_classes = num_classes
        self.standarization = standarization
        self.discretization = discretization
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold
        self.feature_extractor = create_feature_extraction_base(pre_trained_name, feature_layers)
        self.classifier = SVC(random_state=seed)


    def train_classifier(self, train_gen, train_labels, val_gen):
        self.train_features = self.__extract_features(train_gen, is_train=True)
        print('FEATURES!')
        self.classifier.fit(self.train_features, train_labels)
        print('TRAINED!')
        train_predictions = self.classifier.predict(self.train_features)
        val_predictions = self.predict(val_gen)
        return train_predictions, val_predictions


    def predict(self, generator):
        self.test_features = self.__extract_features(generator, is_train=False)
        return self.classifier.predict(self.test_features)


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


    def save_features(self, save_path):
        save_array_to_npy_file(self.train_features, join_path(save_path, 'train_features.npy'))
        save_array_to_npy_file(self.test_features, join_path(save_path, 'val_features.npy'))


    def __extract_features(self, data_gen, is_train):
        features = self.__extract_activations(data_gen)

        if is_train:
            self.__compute_train_mean_std(features)

        if self.standarization:
            features = self.__standarize_features(features)
        if self.discretization:
            features = self.__discretize_features(features)

        return features


    def __standarize_features(self, features):
        standarized_features = np.divide(features - self.train_means, self.train_stds, where=self.train_stds != 0)
        return standarized_features

    
    def __compute_train_mean_std(self, features):
        self.train_means = np.mean(features, axis=0)
        self.train_stds = np.std(features, axis=0)


    def __discretize_features(self, features):
        features[features > self.positive_threshold] = 1
        features[features < self.negative_threshold] = -1
        features[[(features >= self.negative_threshold) & (features <= self.positive_threshold)][0]] = 0
        return features

    
    def __extract_activations(self, data_gen):
        features = []
        for _ in range(len(data_gen)):
            batch_images, _ = next(data_gen)
            features.extend(self.__extract_batch_activations(batch_images))
        return np.array(features)


    def __extract_batch_activations(self, batch_images):
        batch_features = []
        for batch_layer_output in self.feature_extractor(batch_images):
            batch_layer_output = batch_layer_output.numpy()

            if len(batch_layer_output.shape) == 4:
                # spatial average pooling
                batch_layer_output = np.mean(batch_layer_output, axis=(1, 2))

            batch_features.append(batch_layer_output)

        batch_features = np.concatenate(batch_features, axis=-1)
        return batch_features


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


import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from models.pre_trained_models import *
from utils.file_io_utils import *
from utils.constants import *
from utils.path_utils import *
from image_data_generator import *
from utils.csv_utils import *
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score



if __name__ == '__main__':

    experiment_path = join_path(experiments_feature_extraction_path, 'experiment_6')
    batch_size = 32
    use_normalization = True

    train_features = load_npy_file_to_np_array(join_path(experiment_path, 'train_features.npy'))
    val_features = load_npy_file_to_np_array(join_path(experiment_path, 'val_features.npy'))
    
    train_df, val_df = create_train_val_dataframes(data_csv_path, data_path)
    train_gen = create_train_gen_feat_extract(train_df, image_shape_top, batch_size, use_normalization, 'vgg16', 'sparse')
    val_gen = create_val_generator(val_df, image_shape_top, batch_size, use_normalization, 'vgg16', 'sparse')
    val_labels = val_gen.labels
    train_labels = train_gen.labels

    features = np.concatenate([train_features, val_features], axis=0)
    labels = np.concatenate([train_labels, val_labels], axis=0)

    classifier = SVC(random_state=42, C=10, gamma='scale')
    classifier.fit(train_features, train_labels)

    train_predictions = classifier.predict(train_features)
    val_predictions = classifier.predict(val_features)

    train_accuracy = accuracy_score(train_gen.labels, train_predictions)
    val_accuracy = accuracy_score(val_gen.labels, val_predictions)

    print(train_accuracy)
    print(val_accuracy)

    #parameters = {
    #    'C' : [10, 100],
    #    'gamma' : ['scale', 0.1, 1],
    #}
#
    #split = [-1] * train_features.shape[0] + [0] * val_features.shape[0]
    #clf = GridSearchCV(classifier, param_grid=parameters, n_jobs=4, cv=PredefinedSplit(split))
#
    #clf.fit(features, labels)
    #print(clf.best_score_)
    #print(clf.best_params_)
#
    #save_object_to_pkl(clf.best_estimator_, join_path(experiment_path, 'grid_search_clf.pkl'))
    #write_dict_to_json({'best_params': clf.best_params_, 'best_score' : clf.best_score_}, 
    #    join_path(experiment_path, 'grid_search_metrics.json'))
#








"""
pool1_pool
conv2_block3_out
conv3_block4_out
conv4_block4_out
conv5_block3_out

"""
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from models.pre_trained_models import *
from utils.file_io_utils import *
from utils.constants import *
from utils.path_utils import *
from image_data_generator import *
from utils.csv_utils import *
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import LinearSVC, SVC



if __name__ == '__main__':

    experiment_path = join_path(experiments_feature_extraction_path, 'experiment_6')

    train_features = load_npy_file_to_np_array(join_path('./', 'train_features.npy'))
    val_features = load_npy_file_to_np_array(join_path(experiment_path, 'val_features.npy'))
    train_labels = load_npy_file_to_np_array(join_path(experiment_path, 'train_labels.npy'))
    
    train_df, val_df = create_train_val_dataframes(data_csv_path, data_path)
    val_gen = create_val_generator(val_df, (224, 224), 32, True, 'vgg16', 'sparse')
    val_labels = val_gen.labels

    features = np.concatenate([train_features, val_features], axis=0)
    labels = np.concatenate([train_labels, val_labels], axis=0)

    classifier = SVC(tol=0.01)

    parameters = {
        'C' : [0.1, 1],
        'gamma' : ['scale', 0.001]
    }

    split = list(range(train_features.shape[0], train_features.shape[0] + val_features.shape[0]))
    clf = GridSearchCV(classifier, param_grid=parameters, n_jobs=4, cv=PredefinedSplit(split))

    clf.fit(features, labels)
    print(clf.best_score_)
    print(clf.best_params_)








"""
pool1_pool
conv2_block3_out
conv3_block4_out
conv4_block4_out
conv5_block3_out

"""
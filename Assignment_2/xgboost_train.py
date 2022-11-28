from utils.file_io_utils import *
from utils.constants import *
from utils.path_utils import *
from image_data_generator import *
from utils.csv_utils import *
import xgboost as xgb
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

    clf = xgb.XGBClassifier(max_depth=5, objective='multi:softmax')
    clf.fit(train_features, train_labels)
    
    train_pred = clf.predict(train_features)
    val_pred = clf.predict(val_features)
    
    train_acc = accuracy_score(train_labels, train_pred)
    val_acc = accuracy_score(val_labels, val_pred)

    write_dict_to_json({'train_acc': train_acc, 'val_acc' : val_acc}, 
        join_path(experiment_path, 'xgboost_metrics.json'))

    print(f'Train accuracy: {train_acc}')
    print(f'Val accuracy: {val_acc}')








"""
pool1_pool
conv2_block3_out
conv3_block4_out
conv4_block4_out
conv5_block3_out

"""
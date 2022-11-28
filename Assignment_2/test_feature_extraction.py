from types import SimpleNamespace
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.constants import *
from utils.path_utils import *
from utils.csv_utils import create_test_dataframe, get_classes_names
from utils.plot_utils import plot_confusion_matrix
from utils.file_io_utils import load_json_to_dict, load_npy_file_to_np_array
from image_data_generator import create_test_generator
from models.feature_extraction_model import FeatureExtractionModel


def test(experiment_path, test_gen, classes_name, config):
    model = FeatureExtractionModel.load_model(experiment_path, config.seed, 
        config.standarization, config.discretization, config.neg_threshold, 
        config.pos_threshold)
    true_labels = test_gen.labels

    predictions = model.predict(test_gen)

    accuracy = accuracy_score(true_labels, predictions)
    print(f'\nTest accuracy: {accuracy:.2f}\n')
    
    print(classification_report(true_labels, predictions, target_names=classes_name))
    confusion_matrix_ = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(confusion_matrix_, classes_names, experiment_path)



if __name__ == '__main__':
    os.environ['TF_DETERMINISTIC_OPS']  = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    config = load_json_to_dict(join_path(test_feat_ext_model_path, 'hyperparameters.json'))
    config = SimpleNamespace(**config)

    classes_names = get_classes_names(classes_names_path)

    test_df = create_test_dataframe(data_csv_path, data_path)
    test_gen = create_test_generator(test_df, image_shape_top[:-1], 32, config.normalization, config.name, class_mode='sparse')

    test(test_feat_ext_model_path, test_gen, classes_names, config)

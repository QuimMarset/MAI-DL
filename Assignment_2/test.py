from types import SimpleNamespace
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.constants import *
from utils.path_utils import *
from utils.csv_utils import create_test_dataframe, get_classes_names
from utils.plot_utils import plot_confusion_matrix
from utils.file_io_utils import load_mean_std_from_npy, load_json_to_dict
from image_data_generator import create_test_generator
from models.basic_model import BasicModel


def test(experiment_path, test_gen, classes_name):
    model = BasicModel.create_test_model(experiment_path)
    true_labels = test_gen.labels
    predictions = model.predict(test_gen)

    accuracy = accuracy_score(true_labels, predictions)
    print(f'\nTest accuracy: {accuracy:.2f}\n')
    
    print(classification_report(true_labels, predictions, target_names=classes_name))
    confusion_matrix_ = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(confusion_matrix_, classes_names, experiment_path)



if __name__ == '__main__':
    os.environ['TF_DETERMINISTIC_OPS']  = '1'

    parameters = load_json_to_dict(join_path(test_model_path, 'hyperparameters.json'))
    parameters = SimpleNamespace(**parameters)

    train_mean_std = load_mean_std_from_npy(train_mean_std_path)
    classes_names = get_classes_names(classes_names_path)
    
    mean_std = None
    if parameters.train_normalization:
        mean_std = train_mean_std

    test_df = create_test_dataframe(data_csv_path, data_path)
    test_gen = create_test_generator(test_df, image_shape[:-1], 32, mean_std)

    test(parameters, test_model_path, test_gen, image_shape, classes_names)

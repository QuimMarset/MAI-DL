from tensorflow import keras
from sklearn.utils import compute_class_weight
from utils.paths_utils import *
from utils.file_io_utils import *
from utils.plot_utils import *
from utils.train_utils import *




def train(experiments_path, parsed_args, train_gen, val_gen, num_classes):
    experiment_path = create_new_experiment_folder(experiments_path)

    optimizer = create_optimizer(parsed_args)
    model = create_model(parsed_args, num_classes)

    callbacks = []

    if parsed_args.early_stop >= 0:
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=parsed_args.early_stop, 
            restore_best_weights=True)
        callbacks.append(early_stopping)

    class_weights = None
    if parsed_args.balance_classes:
        class_weights = compute_class_weight('balanced', classes=range(num_classes), y=train_gen.get_labels())

    model.compile(optimizer, run_eagerly=True, loss_weights=class_weights)
    history = model.fit(train_gen, epochs=parsed_args.epochs, validation_data=val_gen, workers=4)
    
    model.save_architecture(experiment_path)
    model.save_model_weights(experiment_path)
    # Save training metrics and experiment hyperparameters
    write_dict_to_json(history.history, experiment_path)
    write_dict_to_json(vars(parsed_args), experiment_path, indent=1)


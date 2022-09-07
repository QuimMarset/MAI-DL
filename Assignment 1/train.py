from tensorflow import keras
from utils.paths_utils import *
from utils.file_io_utils import *
from utils.plot_utils import *



def save_train_metrics(metrics_dict, save_path):
    path = os.path.join(save_path, 'metrics.json')
    write_dict_to_json(metrics_dict, path)


def save_hyperparameters(parameters_dict, save_path):
    path = os.path.join(save_path, 'parameters.json')
    write_dict_to_json(parameters_dict, path, indent=1)


def create_optimizer():
    pass


def create_model():
    pass


def train_model(train_gen, val_gen, epochs, learning_rate, image_size, experiment_path, lr_decay=False, workers=4):

    if lr_decay:
        learning_rate = keras.optimizers.schedules.ExponentialDecay(learning_rate, 1000, 0.95, staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)
    model = DepthEstimationModel(ssim_weight, l1_weight, edge_weight, image_size, modified_loss)
    model.compile(optimizer, run_eagerly=True)

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, workers=workers)
    
    model.save_architecture(experiment_path)
    model.save_model_weights(experiment_path)
    save_hyperparameters(experiment_path)
    save_train_metrics(history.history, experiment_path)


def train(experiments_path, params_dict, train_gen, val_gen, lr_decay=False):
    num_experiments = len(os.listdir(experiments_path))
    experiment_path = os.path.join(experiments_path, f'experiment_{num_experiments+1}')
    os.makedirs(experiment_path, exist_ok=True)

    train_model(train_gen, val_gen, params_dict['epochs'], params_dict['lr'], params_dict['image_size'], experiment_path, lr_decay=lr_decay)

    save_hyperparameters(params_dict, experiment_path)



    if weight_loss:
      class_weight = compute_class_weight('balanced', classes=range(10), y=Yt.flatten())
      print(class_weight)
   else:
      class_weight = None

   reset_random_seeds()

   # build network
   model = build_network(codes, 'glove', max_len, suf_len, pref_len, 100, lstm_units, dropout, recurrent_dropout, optimizer, class_weight)
   with redirect_stdout(sys.stderr) :
      model.summary()

   early_stopping = EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=best_weights)
   callbacks = [early_stopping]

   # train model
   with redirect_stdout(sys.stderr) :
      model.fit(Xt, Yt, batch_size=batch_size, epochs=epochs, validation_data=(Xv,Yv), verbose=1, callbacks=callbacks)

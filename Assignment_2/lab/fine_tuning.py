from tensorflow import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os

# Plot the training and validation loss + accuracy
def plot_training(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    #Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation accuracy')
    plt.savefig('fine_tuning_accuracy.pdf')
    plt.close()
    #Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig('fine_tuning_loss.pdf')



img_width, img_height = 256, 256
train_data_dir = "mit67/train"
validation_data_dir = "mit67/test"
nb_train_samples = 5359
nb_validation_samples = 1339 
batch_size = 32
epochs = 50
target_classes = 67

classes = [class_name for class_name in os.listdir(train_data_dir)]

model = keras.applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
for layer in model.layers[:10]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(target_classes, activation="softmax")(x)

# creating the final model 
model_final = Model(model.input, predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
        rescale = 1./255)#,
        #horizontal_flip = True,
        #fill_mode = "nearest",
        #zoom_range = 0.3,
        #width_shift_range = 0.3,
        #height_shift_range=0.3,
        #rotation_range=30)

val_datagen = ImageDataGenerator(
        rescale = 1./255)#,
        #horizontal_flip = True,
        #fill_mode = "nearest",
        #zoom_range = 0.3,
        #width_shift_range = 0.3,
        #height_shift_range=0.3,
        #rotation_range=30)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size, 
        classes=classes,
        class_mode = "categorical")

validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        classes=classes,
        shuffle = False, 
        class_mode = "categorical")

# Save the model according to the conditions  
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model 
history = model_final.fit(
        train_generator,
        epochs = epochs,
        validation_data = validation_generator,
        callbacks = [early])#,checkpoint])

plot_training(history)
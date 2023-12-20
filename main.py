import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, MobileNet, NASNetMobile
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import PIL

PATH = "./DatasetB"
TRAIN_PATH = PATH + '/train_good'
VALID_PATH = PATH + '/valid_good'
TRAIN_DATA = glob.glob(os.path.join(TRAIN_PATH, '*.jpg'))
TRAIN_LABEL = glob.glob(os.path.join(TRAIN_PATH, '*.csv'))
VALID_DATA = glob.glob(os.path.join(VALID_PATH, '*.jpg'))
VALID_LABEL = glob.glob(os.path.join(VALID_PATH, '*.csv'))
print(len(os.listdir(TRAIN_PATH)), len(os.listdir(VALID_PATH)))

dif = [x for x in (os.listdir(TRAIN_PATH)) if x not in (os.listdir(VALID_PATH))]
print(dif)

N_CLASSES = len(os.listdir(TRAIN_PATH))
BATCH_SIZE = 32
IMG_WIDTH, IMG_HEIGHT = 299, 299

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_datagen = ImageDataGenerator(
    rescale=1. / 255)

valid_generator = valid_datagen.flow_from_directory(
    directory=VALID_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

N_TRAIN = train_generator.samples
N_VALID = valid_generator.samples
print(N_TRAIN, N_VALID)

class_indices = train_generator.class_indices
label_list = list(class_indices.keys())
print("Class labels:", label_list)

mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# for layer in mobilenet.layers:
#     layer.trainable = False

# mobilenet
x = mobilenet.output
x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)  # Increased number of neurons
# x = Dropout(0.2)(x)  # Increased dropout rate
x = Dense(128, activation='relu')(x)  # Added an extra Dense layer
x = Dropout(0.5)(x)  # Increased dropout rate
x = Dense(128, activation='relu')(x)  # Added an extra Dense layer
x = Dropout(0.5)(x)  # Increased dropout rate

predictions = Dense(N_CLASSES, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
model = Model(inputs=mobilenet.input, outputs=predictions)

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='mobileNetV4.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('model_v4.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = N_TRAIN // BATCH_SIZE,
                    validation_data = valid_generator,
                    validation_steps=N_VALID // BATCH_SIZE,
                    epochs=10,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('./model_v4.h5')
tflite_model = tf.keras.models.load_model('./model_v4.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
tflite_save = converter.convert()
open("./Model_V4.tflite", "wb").write(tflite_save)
def plot_accuracy(history):

    plt.plot(history.history['accuracy'],label='train accuracy')
    plt.plot(history.history['val_accuracy'],label='validation accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('Accuracy_v4_Model')
    plt.show()

def plot_loss(history):

    plt.plot(history.history['loss'],label="train loss")
    plt.plot(history.history['val_loss'],label="validation loss")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('Loss_v4_Model')
    plt.show()

plot_accuracy(history)
plot_loss(history)
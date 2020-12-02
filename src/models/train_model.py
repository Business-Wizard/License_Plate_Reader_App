import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten,
                                     Conv2D, MaxPooling2D)
from sklearn.metrics import classification_report
import tensorflow as tf
import cv2
from tensorflow.python.ops.gen_io_ops import save
import segmentation
import warnings
warnings.filterwarnings('ignore')
np.random.seed(101)  # for reproducibility

# ! to avoid some common specific hardware/environment errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

data_directory = os.path.join(os.getcwd(), "data")
processed_directory = os.path.join(data_directory, "processed/2_recognition")
holdout_directory = os.path.join(processed_directory, "train_set")
train_directory = os.path.join(processed_directory, "train_set")


def load_images(source: str = train_directory,
                sample_frac: float = 0.01):
    filename_list = os.listdir(source)
    filepath_list = [os.path.join(train_directory, file)
                     for file in filename_list]
    dataset_size = int(len(filepath_list) * sample_frac)
    # images_array = np.empty((dataset_size, 7, 30, 30), dtype=np.float32)
    images_array = np.empty((dataset_size*7, 30, 30), dtype=np.float32)
    # ! dataset_size * 7  for testing different method
    print(images_array.shape)
    print("LOADING IMAGES ARRAY...")

    # for idx, filepath in zip(range(dataset_size), filepath_list):
    #     segments = segmentation.segment_image(cv2.imread(filepath, 0))
    #     images_array[idx] = segments

    for idx, filepath in zip(range(0, dataset_size*7, 7), filepath_list):
        segments = segmentation.segment_image(cv2.imread(filepath, 0))
        images_array[idx], images_array[idx+1], images_array[idx+2], images_array[idx+3], images_array[idx+4], images_array[idx+5], images_array[idx+6] = segments

    print("DONE LOADING IMAGES")
    return images_array


def load_labels(source: str = train_directory,
                sample_frac: float = 0.01):
    filename_list = os.listdir(source)

    labels_list = [file[-11:-4].lower().replace('-', '')
                   for file in filename_list]
    dataset_size = int(len(labels_list) * sample_frac)
    plate_number_length = 7
    labels_array = np.empty((dataset_size, plate_number_length), dtype='U10')
    # print("LOADING LABELS ARRAY...")
    # for idx1, label in zip(range(dataset_size), labels_list):
    #     for idx2, char in enumerate(label):
    #         labels_array[idx1, idx2] = char

    labels_array = np.empty((dataset_size*7, 1), dtype='U10')
    print("LOADING LABELS ARRAY...")
    for idx1, label in zip(range(0, dataset_size*7, 7), labels_list):
        for iter, char in enumerate(label):
            labels_array[idx1+iter] = char

    print("DONE LOADING LABELS")
    return labels_array


def standardize_data(X, y, image_shape: tuple = (30, 30)):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # reshape input into format Conv2D layer likes
    X_train = X_train.reshape(X_train.shape[0],
                                image_shape[0], image_shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0],
                            image_shape[0], image_shape[1], 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, X_test, y_train, y_test


def categorical_encoding(y):
    encoder = OneHotEncoder(handle_unknown='error', sparse=False)
    encoder.fit(y)
    print(encoder.categories_)
    labels_array = encoder.transform(y)
    return encoder, labels_array


def load_data(source: str = train_directory,
              sample_frac: float = 0.01):

    images_array = load_images(source, sample_frac)
    labels_array = load_labels(source, sample_frac)
    # labels_array = to_categorical(labels_array, num_classes=33)
    encoder, labels_array = categorical_encoding(labels_array)

    X_train, X_test, y_train, y_test =\
        standardize_data(images_array, labels_array, image_shape=(30, 30))

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrice (don't change)
    # Y_train = to_categorical(y_train, num_classes)
    # Y_test = to_categorical(y_test, num_classes)
    # in Ipython you should compare Y_test to y_test
    return X_train, X_test, \
        y_train, y_test, encoder


def define_model(num_filters, kernel_size, input_shape, pool_size,
                 num_classes: int = 33):
    model = Sequential()  # model is a linear stack of layers (don't change)
    # note: convolutional and dense layers require an activation function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    model.add(Conv2D(num_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid', input_shape=input_shape,
                     activation='relu'))  # ! 1st conv. layer
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    # ! necessary to flatten before going into conventional dense layer
    print('Model flattened out to ', model.output_shape)

    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # see https://keras.io/optimizers/#usage-of-optimizers
    # * KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # * and KEEP metrics at 'accuracy'
    # ? optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def save_model(model, destination: str = "./models/",
               filename: str = "model"):
    print("SAVING MODEL")
    tf.keras.models.save_model(model,
                               os.path.join(destination, (filename + "_full")),
                               include_optimizer=True, save_format="h5")
    model.save_weights(os.path.join(destination, (filename + "_weights")),
                       save_format="h5")


def visualize_history(model):
    for key in model_history.history.keys():
        plt.plot(model_history.history[key])
        plt.title(key)
        plt.xlabel('epoch')
        plt.ylabel(key)
        plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, encoder =\
        load_data(train_directory, sample_frac=0.1)
    print("DATA READY")

    model = define_model(num_filters=40,
                         kernel_size=(4, 4), input_shape=(30, 30, 1),
                         pool_size=(2, 2), num_classes=33)
    print(model.summary())

    model_history = model.fit(X_train, y_train, batch_size=15,
                              epochs=3, verbose=1,
                              validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    # predictions = model.predict(X_test)  #one-hot encoded
    # predictions = model.predict_classes(X_test).reshape((-1, 1)) #deprecated
    predictions = np.argmax(model.predict(X_test),
                            axis=-1).reshape((-1, 1))
    y_test = np.argmax(y_test, axis=-1).reshape((-1, 1))
    print(predictions[0:10])
    print(predictions.shape)
    print("####################################################")
    print(y_test[0:10])
    print(y_test.shape)
    # y_test = encoder.inverse_transform(predictions)
    # print(y_test[0:10])
    # print(y_test.shape)

    print(classification_report(y_test, predictions))
    print(f'Test score: {score[0]}')
    print(f'Test accuracy: {score[1]}')  # this is the one we care about

    # visualize_history(model)

    save_model(model)

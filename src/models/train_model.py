import os
import numpy as np
from pandas.core.arrays import string_
from tensorflow.python.ops.gen_io_ops import tf_record_reader_v2_eager_fallback
from tensorflow.python.ops.gen_math_ops import imag_eager_fallback
import segmentation
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten,
                                     Conv2D, MaxPooling2D)
from sklearn.metrics import classification_report
# import tensorflow as tf
import cv2

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


def load_data(source: str = train_directory,
              sample_frac: float = 0.01):

    images_array = load_images(source, sample_frac)
    labels_array = load_labels(source, sample_frac)
    # labels_array = to_categorical(labels_array, num_classes=33)
    encoder = OneHotEncoder(handle_unknown='error', sparse=False)
    encoder.fit(labels_array)
    print(encoder.categories_)
    labels_array = encoder.transform(labels_array)

    X_train, X_test, y_train, y_test =\
        standardize_data(images_array, labels_array, image_shape=(30, 30))

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrice (don't change)
    # Y_train = to_categorical(y_train, nb_classes)
    # Y_test = to_categorical(y_test, nb_classes)
    # in Ipython you should compare Y_test to y_test
    return X_train, X_test, y_train, y_test


def define_model(nb_filters, kernel_size, input_shape, pool_size, nb_classes: int = 33):
    model = Sequential()  # model is a linear stack of layers (don't change)
    # note: convolutional and dense layers require an activation function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='same', input_shape=input_shape,
                     activation='relu'))  # ! 1st conv. layer
    # model.add(Activation('relu'))
    # ! Activation specification necessary for Conv2D and Dense layers
    # model.add(Conv2D(nb_filters,
    #                  (kernel_size[0], kernel_size[1]),
    #                  padding='same', input_shape=input_shape,
    #                  activation='relu'))  # ! 2nd conv. layer
    # model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    # decreases size, helps prevent overfitting

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='same', input_shape=input_shape,
                     activation='relu'))  # ! 3rd conv. layer
    # model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    # decreases size, helps prevent overfitting

    model.add(Flatten())
    # ! necessary to flatten before going into conventional dense layer
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(40))
    # ! (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))
    model.add(Dense(40))
    # ! (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    model.add(Dense(nb_classes, activation='softmax'))
    # ! 10 final nodes (one for each class)
    # ! softmax at end to pick between classes 0-33 

    # see https://keras.io/optimizers/#usage-of-optimizers
    # * KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # * and KEEP metrics at 'accuracy'
    # ? optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# TODO character-level accuracy for evaluation


if __name__ == '__main__':
    X_train, X_test, y_train, y_test =\
        load_data(train_directory, sample_frac=0.01)
    print("DATA READY")

    # nb_classes = 10  # ! number of output possibilities: [0 - 9] KEEP
    # passes through the entire train dataset before weights "final"
    # img_rows, img_cols = 28, 28   # ! the size of the MNIST images KEEP
    # input_shape = (img_rows, img_cols, 1)
    # ! 1 channel image input (grayscale) KEEP
    # nb_filters = 15  # number of convolutional filters to use
    # pool_size = (2, 2)
    # decrease img size, reduce computatn, adds translational invariance
    # kernel_size = (3, 3)
    # convolutional kernel size, slides over image to learn features

    model = define_model(nb_filters=32,
                         kernel_size=(4, 4), input_shape=(30, 30, 1),
                         pool_size=(2, 2), nb_classes=33)
    print(model.summary())

    batch_size = 100
    # number of training samples used at a time to update the weights
    nb_epoch = 3

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, y_test))
    print(model.metrics_names)
    score = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict_classes(X_test)
    print(classification_report(y_test, predictions))

    print(f'Test score: {score[0]}')
    print(f'Test accuracy: {score[1]}')  # this is the one we care about

    '''single save of model + weights'''
    # model.save("../../models")

    '''save architecture and weights separately'''
    # serialize model to disk
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)

    # serialize weights to HDF5
    # model.save_weights("model.h5")
    # print("Saved weights to disk")

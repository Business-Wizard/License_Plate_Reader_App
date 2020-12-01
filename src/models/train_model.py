import os
import numpy as np
from tensorflow.python.ops.gen_math_ops import imag_eager_fallback
import segmentation
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten,
                                     Conv2D, MaxPooling2D)
import cv2

data_directory = os.path.join(os.getcwd(), "data")
processed_directory = os.path.join(data_directory, "processed/2_recognition")
train_directory = os.path.join(processed_directory, "train_set")

def load_images(source: str = train_directory,
              sample_frac: float = 0.01):
    filename_list = os.listdir(source)
    filepath_list = [os.path.join(train_directory, file)
                     for file in filename_list]
    images_array = np.empty((100000, 7, 30, 30), dtype=np.float32)
    images_array.fill(0.25)
    print("LOADING ARRAY...")

    dataset_size = int(len(filepath_list) * sample_frac)
    for idx, filepath in zip(range(dataset_size), filepath_list):
        segments = segmentation.segment_image(cv2.imread(filepath, 0))
        images_array[idx] = segments

    for idx, filepath in enumerate(filepath_list):
        segments = segmentation.segment_image(cv2.imread(filepath, 0))
        images_array[idx] = segments

    print("DONE LOADING")
    return images_array


def load_labels(source: str = train_directory,
                sample_frac: float = 0.01):
    filename_list = os.listdir(source)
    labels_list = [file[-11:-4].lower().replace('-', '')
                   for file in filename_list]
    dataset_size = len(labels_list)
    plate_number_length = 7
    labels_array = np.empty((dataset_size, plate_number_length), dtype='U12')
    print("LOADING ARRAY...")
    for idx1, label in enumerate(labels_list):
        for idx2, char in enumerate(label):
            labels_array[idx1, idx2] = char
            print(labels_array[idx1, idx2])
    print("DONE LOADING")
    return labels_array


def load_data(source: str = train_directory,
                        sample_frac: float = 0.01):
    images_array = load_images(source, sample_frac)
    labels_array = load_labels(source, sample_frac)
    return images_array, labels_array


    # segment train images
    # X_train, X_test, Y_train, Y_test = load_and_featurize_data()
    # load segments into X array
    # slice image name for labels
    # load labels into y array
    # to_categorical


def next_steps():
    # split segments into train/test
    # the data, shuffled and split between train and test sets
    # X_train, y_train, X_test, y_test =
    # reshape input into format Conv2D layer likes
    # X_train = X_train.reshape(X_train.shape[0],
    #           image_shape[0], image_shape[1], 1)
    # X_test = X_test.reshape(X_test.shape[0],
    #          image_shape[0], image_shape[1], 1)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices (don't change)
    # Y_train = to_categorical(y_train, nb_classes)
    # Y_test = to_categorical(y_test, nb_classes)
    # in Ipython you should compare Y_test to y_test

    # TODO
    # * for setting the 36 classes to categorical
    # * y_cat_train = to_vategorical(y_test, 36)
    # return X_train, X_test, Y_train, Y_test
    pass


def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential()  # model is a linear stack of layers (don't change)
    # note: convolutional and dense layers require an activation function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='same',
                     input_shape=input_shape))  # ! first conv. layer KEEP
    model.add(Activation('relu'))
    # Activation specification necessary for Conv2D and Dense layers
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='same',
                     input_shape=input_shape))  # ! 2nd conv. layer KEEP
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    # decreases size, helps prevent overfitting

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='same',
                     input_shape=input_shape))  # ! 3rd conv. layer  KEEP
    model.add(Activation('relu'))
    # Activation specification necessary for Conv2D and Dense layers

    model.add(MaxPooling2D(pool_size=pool_size))
    # decreases size, helps prevent overfitting

    model.add(Flatten())
    # ! necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(40))
    # ! (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))
    model.add(Dense(40))
    # ! (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    model.add(Dense(nb_classes))
    # ! 10 final nodes (one for each class)  KEEP
    model.add(Activation('softmax'))
    # ! softmax at end to pick between classes 0-9 KEEP

    # see https://keras.io/optimizers/#usage-of-optimizers
    # * KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # * and KEEP metrics at 'accuracy'
    # ? suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# TODO train/test split for tuning

# TODO character-level accuracy for evaluation

if __name__ == '__main__':
    X, y = load_data(train_directory, sample_frac=0.0001)

    # ! important inputs to the model: don't changes the ones marked KEEP
    # batch_size = 5000
    # number of training samples used at a time to update the weights
    # nb_classes = 10  # ! number of output possibilities: [0 - 9] KEEP
    # nb_epoch = 30
    # passes through the entire train dataset before weights "final"
    # img_rows, img_cols = 28, 28   # ! the size of the MNIST images KEEP
    # input_shape = (img_rows, img_cols, 1)
    # # ! 1 channel image input (grayscale) KEEP
    # nb_filters = 15   # number of convolutional filters to use
    # pool_size = (2, 2)
    # decrease image size, reduce computatn, adds translational invariance
    # kernel_size = (3, 3)
    # # convolutional kernel size, slides over image to learn features

    # X_train, X_test, Y_train, Y_test = load_and_featurize_data()
    # model = define_model(nb_filters, kernel_size, input_shape, pool_size)
    # during fit process watch train and test error simultaneously
    # model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
    #           verbose=1, validation_data=(X_test, Y_test))

    # evaluate model on train data
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print(f'Test accuracy:, {score[1]}')  # this is the one we care about

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

    print("COMPLETE")

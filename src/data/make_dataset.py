import os
from src.features.build_features import read_image
from tensorflow.keras.utils import to_categorical
import numpy as np
from src.features import *

def list_directory(directory: str):
    return [dir.lower().replace('-', '').split('.')[0] for dir in os.listdir(directory)]

#TODO create data structure to hold snip image data
#TODO decide on shape of data structure
def create_data_array():
    np.array()
    pass

def load_and_featurize_data(image_shape: tuple=(30,30)):
    # the data, shuffled and split between train and test sets
    # X_train, y_train, X_test, y_test = 
    # reshape input into format Conv2D layer likes
    X_train = X_train.reshape(X_train.shape[0], image_shape[0], image_shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], image_shape[0], image_shape[1], 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices (don't change)
    Y_train = to_categorical(y_train, nb_classes)  # cool
    Y_test = to_categorical(y_test, nb_classes)
    # in Ipython you should compare Y_test to y_test
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    current_working_directory = os.getcwd()
    #! update to correct data folder holding your license plate images
    data_directory = current_working_directory + "/data/external/2_recognition/license_synthetic/license-plates"
    data_lst = list_directory(data_directory)
    print(data_lst[0])
    






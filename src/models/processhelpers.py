import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import cv2
import segmentation
import warnings
warnings.filterwarnings('ignore')
np.random.seed(101)  # for reproducibility

data_directory = os.path.join(os.getcwd(), "data")
processed_directory = os.path.join(data_directory, "processed/2_recognition")
holdout_directory = os.path.join(processed_directory, "holdout_set")
train_directory = os.path.join(processed_directory, "train_set")
prediction_directory = os.path.join(data_directory, "processed/3_prediction/")


def load_images(source: str = train_directory,
                sample_frac: float = 0.01):
    filename_list = os.listdir(source)
    filepath_list = [os.path.join(source, file)
                     for file in filename_list]
    dataset_size = int(len(filepath_list) * sample_frac)
    dataset_size = dataset_size if dataset_size >= 1 else 1
    # images_array = np.empty((dataset_size, 7, 30, 30), dtype=np.float32)
    images_array = np.empty((dataset_size*7, 30, 30), dtype=np.float32)
    print(images_array.shape)
    print("LOADING IMAGES ARRAY...")

    for idx, filepath in zip(range(0, dataset_size*7, 7), filepath_list):
        segments = segmentation.segment_image(cv2.imread(filepath, 0))
        images_array[idx], images_array[idx+1], images_array[idx+2],
        images_array[idx+3], images_array[idx+4], images_array[idx+5],
        images_array[idx+6] = segments

    print("DONE LOADING IMAGES")
    return images_array


def load_labels(source: str = train_directory,
                sample_frac: float = 0.01):
    filename_list = os.listdir(source)

    labels_list = [file[-11:-4].lower().replace('-', '')
                   for file in filename_list]
    dataset_size = int(len(labels_list) * sample_frac)
    dataset_size = dataset_size if dataset_size >= 1 else 1
    plate_number_length = 7
    labels_array = np.empty((dataset_size, plate_number_length), dtype='U10')

    labels_array = np.empty((dataset_size*7, 1), dtype='U10')
    print("LOADING LABELS ARRAY...")
    for idx1, label in zip(range(0, dataset_size*7, 7), labels_list):
        for iter, char in enumerate(label):
            labels_array[idx1+iter] = char

    print("DONE LOADING LABELS")
    return labels_array


def standardize_data(X, image_shape: tuple = (30, 30)):
    # reshape input into format Conv2D layer likes
    X = X.reshape(X.shape[0], image_shape[0], image_shape[1], 1)
    X = X.astype('float32')
    X /= 255
    return X


def categorical_encoding(y):
    encoder = OneHotEncoder(handle_unknown='error', sparse=False)
    encoder.fit(y)
    print(f"Categories: {encoder.categories_}")
    labels_array = encoder.transform(y)
    return encoder, labels_array


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def load_test_data(source: str = train_directory,
                   sample_frac: float = 0.01, validate: bool = False):

    images_array = load_images(source, sample_frac)
    labels_array = load_labels(source, sample_frac)
    # labels_array = to_categorical(labels_array, num_classes=33)
    encoder, labels_array = categorical_encoding(labels_array)
    X, y = standardize_data(images_array, labels_array, image_shape=(30, 30))

    if validate:
        return X, y, encoder

    else:
        X_train, X_test, y_train, y_test = split_data(X, y)
        print('X_train shape:', X_train.shape)
        print('y_train shape:', y_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        return X_train, X_test, \
            y_train, y_test, encoder


def load_unseen_data(source: str = prediction_directory,
                     sample_frac: float = 1.0):
    images_array = load_images(source, sample_frac)
    X = standardize_data(images_array, image_shape=(30, 30))
    return X


def zip_prediction_labels(predictions_array, encoder, plate_length: int = 7):
    predictions_size = len(predictions_array) // plate_length
    prediction_labels = np.empty((predictions_size), dtype='U10')
    predictions = encoder.inverse_transform(predictions_array).flatten()
    print(predictions.shape)

    for idx in range(predictions_size):
        idx2 = idx * plate_length
        chars_lst =\
            [char for __, char in zip(
                range(plate_length), predictions[idx2:idx2+plate_length]
                )]
        label = "".join(chars_lst)
        # print(f'label: {label}')
        prediction_labels[idx] = label
        # print(f"Predicted: {prediction_labels[idx]}")

    return prediction_labels

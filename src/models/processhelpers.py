import logging
import os
import pathlib
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Final

import string

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.models import segmentation

warnings.filterwarnings('ignore')
np.random.seed(101)  # for reproducibility

DATA_DIR: Final[Path] = pathlib.Path().cwd() / 'data'
PROCESSED_DIR: Final[Path] = DATA_DIR / 'processed/2_recognition'
HOLDOUT_DIR: Final[Path] = PROCESSED_DIR / 'holdout_set'
TRAIN_DIR: Final[Path] = PROCESSED_DIR / 'train_set'
PREDICTION_DIR: Final[Path] = DATA_DIR / 'processed/3_prediction/'

encoder = OneHotEncoder(handle_unknown='error', sparse=False)
license_plate_letters: Final[str] = (
    string.ascii_lowercase.replace('o', '').replace('q', '').replace('z', '')
)
license_plate_characters: Final[str] = string.digits + license_plate_letters
categories_array = np.array(
    object=list(license_plate_characters),
    dtype='<U5',
).reshape((-1, 1))
encoder.fit(categories_array)


def load_images(source: Path = TRAIN_DIR, sample_frac: float = 0.01):
    filename_list: Iterable[Path] = source.iterdir()
    filepath_list = [os.path.join(source, file) for file in filename_list]
    dataset_size = int(len(filepath_list) * sample_frac)
    dataset_size = dataset_size if dataset_size >= 1 else 1
    images_array = np.empty((dataset_size * 7, 30, 30), dtype=np.float32)
    logging.info(f'Source Images Shape: {images_array.shape}')
    logging.info('LOADING IMAGES ARRAY...')

    for idx, filepath in zip(range(0, dataset_size * 7, 7), filepath_list):
        segments = segmentation.segment_image(cv2.imread(filepath, 0))
        (
            images_array[idx],
            images_array[idx + 1],
            images_array[idx + 2],
            images_array[idx + 3],
            images_array[idx + 4],
            images_array[idx + 5],
            images_array[idx + 6],
        ) = segments

    logging.info('DONE LOADING IMAGES')
    return images_array


def load_single_image(source):
    images_array = np.empty((7, 30, 30), dtype=np.float32)
    logging.info(f'Source Images Shape: {images_array.shape}')
    logging.info('LOADING IMAGES ARRAY...')
    segments = segmentation.segment_image(source)
    (
        images_array[0],
        images_array[1],
        images_array[2],
        images_array[3],
        images_array[4],
        images_array[5],
        images_array[6],
    ) = segments
    logging.info('DONE LOADING IMAGES')
    return images_array


def load_labels(source: str = TRAIN_DIR, sample_frac: float = 0.01):
    filename_list = os.listdir(source)

    labels_list = [file[-11:-4].lower().replace('-', '') for file in filename_list]
    dataset_size = int(len(labels_list) * sample_frac)
    dataset_size = dataset_size if dataset_size >= 1 else 1
    plate_number_length = 7
    labels_array = np.empty((dataset_size, plate_number_length), dtype='U10')

    labels_array = np.empty((dataset_size * 7, 1), dtype='U10')
    logging.info('LOADING LABELS ARRAY...')
    for idx1, label in zip(range(0, dataset_size * 7, 7), labels_list, strict=True):
        for iter, char in enumerate(label):
            labels_array[idx1 + iter] = char

    logging.info('DONE LOADING LABELS')
    return labels_array


def standardize_data(X, image_shape: tuple = (30, 30)):
    # reshape input into format Conv2D layer likes
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = X.astype('float32')
    X /= 255
    return X


def categorical_encoding(y):
    encoder = OneHotEncoder(handle_unknown='error', sparse=False)
    encoder.fit(y)
    logging.info(f'Categories: {encoder.categories_}')
    labels_array = encoder.transform(y)
    return encoder, labels_array


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return x_train, x_test, y_train, y_test


def load_test_data(
    source: str = TRAIN_DIR,
    sample_frac: float = 0.01,
    validate: bool = False,
):
    images_array = load_images(source, sample_frac)
    labels_array = load_labels(source, sample_frac)
    encoder, labels_array = categorical_encoding(labels_array)
    X = standardize_data(images_array, labels_array)
    y = labels_array

    if validate:
        return X, y, encoder

    else:
        X_train, X_test, y_train, y_test = split_data(X, y)
        logging.info('X_train shape:', X_train.shape)
        logging.info('y_train shape:', y_train.shape)
        logging.info(X_train.shape[0], 'train samples')
        logging.info(X_test.shape[0], 'test samples')

        return X_train, X_test, y_train, y_test, encoder


def load_unseen_data(source: str = PREDICTION_DIR, sample_frac: float = 1.0):
    images_array = load_images(source, sample_frac)
    X = standardize_data(images_array, image_shape=(30, 30))
    return X


def zip_prediction_labels(predictions_array: np.ndarray, encoder, plate_length: int = 7):
    predictions_size: int = len(predictions_array) // plate_length
    prediction_labels = np.empty((predictions_size), dtype='U10')
    predictions = encoder.inverse_transform(predictions_array).flatten()

    for idx in range(predictions_size):
        idx2 = idx * plate_length
        chars_lst = [
            char for __, char in zip(range(plate_length), predictions[idx2 : idx2 + plate_length])
        ]
        label: str = ''.join(chars_lst)
        prediction_labels[idx] = label
    logging.info(f'Predictions size: {prediction_labels.shape}')

    return prediction_labels

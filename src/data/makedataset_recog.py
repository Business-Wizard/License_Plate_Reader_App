import logging
import os
import pathlib
import shutil
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from src.data.imageprocess import pipeline_single

DATA_DIR: Path = pathlib.Path().cwd() / 'data'
DEFAULT_IMAGE_DIR: Path = DATA_DIR / 'external/2_recognition' / 'license_synthetic/license-plates'
DEFAULT_INTERIM_DIR: Path = DATA_DIR / 'interim/2_recognition/'
DEFAULT_PROCESSED_DIR: Path = DATA_DIR / 'processed/2_recognition/'


def get_image_labels(directory: Path) -> Iterable[str]:
    files: Iterable[Path] = directory.iterdir()
    label_names: Iterable[str] = (filepath.name for filepath in files)
    return (label.lower().replace('-', '') for label in label_names)


def process_directory(
    directory_input: Path = DEFAULT_IMAGE_DIR,
    directory_output: Path = DEFAULT_INTERIM_DIR,
    size: int = 10,
):
    image_labels = get_image_labels(directory_input)
    orig_images = os.listdir(directory_input)
    if size == -1:
        size = 1000000
    for image, label, __ in zip(orig_images, image_labels, range(size), strict=False):
        old_image_name: Path = directory_input / image
        new_image_name: Path = directory_output / f'{label}.png'
        processed_image = pipeline_single(
            old_image_name,
            dilatekernel=(3, 3),
            blurkernel=(5, 5),
            div=15,
            gauss=True,
        )
        cv2.imwrite(str(new_image_name), processed_image)


def holdout_split_directory(
    directory_input: Path = DEFAULT_INTERIM_DIR,
    directory_output: Path = DEFAULT_PROCESSED_DIR,
    test_size: float = 0.3,
):
    filenames = np.array(os.listdir(directory_input))
    np.random.shuffle(filenames)

    split_idx = int(len(filenames) * test_size)
    holdout_set = filenames[:split_idx]
    train_set = filenames[split_idx:]

    logging.info('creating train set')
    for filename in train_set:
        source: Path = directory_input / filename
        destination: Path = directory_output / 'train_set' / filename
        shutil.copy(source, destination)
    logging.info('creating holdout set')
    for filename in holdout_set:
        source: Path = directory_input / filename
        destination: Path = directory_output / 'holdout_set' / filename
        shutil.copy(source, destination)
    logging.info('data sets creation: Done')
    # X_train, X_holdout, y_train, y_holdout = train_test_split(images_array
    #     , label_names, test_size=0.3, random_state=101)


if __name__ == '__main__':
    '''get data directory'''
    # ! update to correct data folder holding your license plate images
    image_directory = DEFAULT_IMAGE_DIR

    '''save processed images with label as filename'''
    # ! uncomment and run to process designated folder of images
    # process_directory(directory_input=default_image_directory,
    #                   directory_output=default_interim_directory, size=-1)

    '''split into holdout and train sets'''
    # ! uncomment to create holdout and train splits
    # holdout_split_directory(directory_input=default_interim_directory,
    #                         test_size=0.3)

    # ! used to correct a single image
    # processed_image = pipeline_single(old_image_name, dilatekernel=(3,3),
    #                                   blurkernel=(5,5), div=25, gauss=True)

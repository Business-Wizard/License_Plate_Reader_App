from os.path import split
from imageprocess import pipeline_single
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

data_directory = os.path.join(os.getcwd(), "data")
default_image_directory = os.path.join(data_directory, "external/2_recognition/license_synthetic/license-plates")
default_interim_directory = os.path.join(data_directory, "interim/2_recognition/")
default_processed_directory = os.path.join(data_directory, "processed/2_recognition/")

def get_image_labels(directory: str):
    return [dir.lower().replace('-', '').split('.')[0] for dir in os.listdir(directory)]

def process_directory(directory_output: str=default_interim_directory
    ,directory_input: str=default_image_directory
    ,size: int=10):
    
    image_labels = get_image_labels(directory_input)
    orig_images = os.listdir(directory_input)
    if size == -1:
        size = 1000000
    for image, label, __ in zip(orig_images, image_labels, range(size)):
        old_image_name = os.path.join(directory_input, image)
        new_image_name = directory_output + label + ".png"
        processed_image = pipeline_single(old_image_name, dilatekernel=(3,3),blurkernel=(5,5), div=25, gauss=True)
        cv2.imwrite(new_image_name, processed_image)

def holdout_split_directory(directory_input: str=default_interim_directory
    , directory_output: str=default_processed_directory
    , test_size: float=0.3):

    filenames = np.array(os.listdir(directory_input))
    np.random.shuffle(filenames)
    
    split_idx = int(len(filenames) * test_size)
    train_set = filenames[:split_idx]
    holdout_set = filenames[split_idx:]

    print("TRAIN SET...")
    for filename in train_set:
        source = os.path.join(directory_input, filename)
        destination = os.path.join(directory_output, "train_set", filename)
        shutil.copy(source, destination)
    print("HOLDOUT SET...")
    for filename in holdout_set:
        source = os.path.join(directory_input, filename)
        destination = os.path.join(directory_output, "holdout_set", filename)
        shutil.copy(source, destination)
    
    print("DONE")

    label_names = [label.replace(".png", "")for label in filenames]
    # X_train, X_holdout, y_train, y_holdout = train_test_split(images_array
    #     , label_names, test_size=0.3, random_state=101)

    return None


if __name__ == '__main__':
    '''get data directory'''
    #! update to correct data folder holding your license plate images
    image_directory = default_image_directory

    '''save processed images with label as filename'''
    #! uncomment and run to process designated folder of images
    # process_directory(directory_input = default_image_directory,
    # directory_output=default_interim_directory, size=1)

    '''split into holdout and train sets'''
    #! uncomment to create holdout and train splits
    # holdout_split_directory(directory_input=default_interim_directory, directory_output=default_processed_directory, test_size=0.3)


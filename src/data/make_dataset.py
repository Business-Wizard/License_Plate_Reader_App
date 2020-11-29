from src.features import build_features
import os
import numpy as np
import cv2

def get_image_labels(directory: str):
    return [dir.lower().replace('-', '').split('.')[0] for dir in os.listdir(directory)]

#TODO create data structure to hold snip image data
#TODO decide on shape of data structure
def read_images():
    np.array()
    pass

#TODO save file with name of image

'''
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
'''

if __name__ == '__main__':
    '''get data directory'''
    current_working_directory = os.getcwd()
    #! update to correct data folder holding your license plate images
    image = current_working_directory + "/data/external/2_recognition/license_synthetic/license-plates"

    '''get image labels'''
    orig_images = os.listdir(image)
    image_labels = get_image_labels(image)
    # print(image_labels[0])
    directory_to_save_images = "../../data/interim/"
    print(len(image_labels))
        
    '''save processed images with label as filename'''
    for image, label in zip(orig_images, image_labels):
        new_image_name = directory_to_save_images + label + ".png"
        character_segments = build_features.pipeline_single(image, dilatekernel=(3,3),blurkernel=(5,5), div=25, gauss=True)
        for idx, segment in enumerate(character_segments):
            segment_name = directory_to_save_images + label[idx] + ".png"
            cv2.imwrite(segment_name, segment)
    









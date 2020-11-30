from src.data.imageprocess import pipeline_single
import os
import cv2
import numpy as np

data_directory = os.path.join(os.getcwd(), "data/external/2_recognition/license_synthetic/license-plates/data/")
default_image_directory = os.path.join(os.getcwd(), "external/2_recognition/")
default_interim_directory = os.path.join(os.getcwd(), "interim/2_recognition/")
default_processed_directory = os.path.join(os.getcwd(), "processed/2_recognition/")

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
        old_image_name = directory_input + image
        new_image_name = directory_output + label + ".png"
        processed_image = pipeline_single(old_image_name, dilatekernel=(3,3),blurkernel=(5,5), div=25, gauss=True)
        cv2.imwrite(new_image_name, processed_image)

def split_train_test(directory_input: str=default_interim_directory
    , directory_output: str=default_processed_directory
    , test_size: float=0.3):
    
    filenames = os.listdir(directory_input)
    images_array = np.array()

    for filename in filenames:
        filepath = os.path.join(default_interim_directory, filename)
        image = cv2.imread(filepath)
        images_array = images_array.vstack(image)
        # fill array np.vstack
    np.random.shuffle(images_array)
    # shuffle array np.random.shuffle

    # train/test array split

    # mkdir train/, test/  os.mkdir

    # copy files os.

    images = os.listdir(directory_input)

    

#TODO create data structure to hold snip image data
#TODO decide on shape of data structure
def read_images():
    np.array()
    pass

#TODO save file with name of image
#! will likely need to save 100,000 images instead of the ~600,000 snips
#! due to name collisions and lack of variety in snips

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
    image_directory = current_working_directory + "/data/external/2_recognition/license_synthetic/license-plates/"

    '''get image labels'''
    orig_images = os.listdir(image_directory)
    image_labels = get_image_labels(image_directory)
    directory_to_save_images = "../../data/interim/"
    print(len(image_labels))
    directory_to_save_images = os.path.join(os.getcwd(), "data", "interim", "2_recognition/")
    
    '''save processed images with label as filename'''
    # process_directory(directory_output=directory_to_save_images, size=10)
    
    # new_image_name = directory_to_save_images + image_labels[0] + ".png"
    # print(f'input image name: {orig_images[0]}')
    # print(f'new image name: {new_image_name}')
    # image_input = image_directory + orig_images[0]
    # print(image_input)
    # character_segments = pipeline_single(image_input, dilatekernel=(3,3),blurkernel=(5,5), div=25, gauss=True)








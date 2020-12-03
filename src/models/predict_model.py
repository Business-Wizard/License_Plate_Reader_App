from processhelpers import holdout_directory, load_data, zip_prediction_labels
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

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


def load_model_with_weights(model_name: str = "model_full",
                            directory: str = "./models/"):
    model = load_model(directory + model_name)
    return model


def load_data_to_predict(source, sample_frac: float = 1.0):
    images_array, labels_array, encoder =\
         load_data(holdout_directory,
                   sample_frac=0.01, predict=True)
    print(f"Images Shape: {images_array.shape}")
    print(f"Labels Shape: {labels_array.shape}")
    return images_array, labels_array, encoder


#TODO apply pipeline on new/holdout image(s)
#! only use if a new image that was not processed!
# return character segment snips


#TODO predict full string of character snips for single image
# plate_predicted = list()
# for char in chars_list:
#    plate_predicted.append(model.predict(char))
#return plate_predicted

#TODO predict, for folder of images
#! apply pipeline only to unseen images!
#return


# TODO license-plate level accuracy for evaluation


if __name__ == '__main__':
    model = load_model_with_weights("model_full")
    X, y, encoder = load_data_to_predict(holdout_directory, sample_frac=0.5)

    score = model.evaluate(X, y, verbose=0)
    print('Test score:', score[0])
    print(f'Test accuracy:, {score[1]}')  # this is the one we care about

    predictions_array = model.predict(X)
    predictions = np.argmax(predictions_array, axis=-1).reshape(-1, 1)
    predictions_labeled = zip_prediction_labels(predictions_array, encoder)
    print(predictions_labeled[0:3])




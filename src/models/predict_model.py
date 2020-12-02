from processhelpers import holdout_directory, load_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import processhelpers


def load_model_with_weights(model_name: str = "model_full",
                            directory: str = "./models/"):
    model = load_model(directory + model_name)
    return model


def load_data_to_predict(source, sample_frac: float = 1.0):
    images_array, labels_array =\
         processhelpers.load_data(holdout_directory,
                                  sample_frac=0.01, predict=True)
    return X, y

#TODO apply pipeline on new/holdout image(s)
#! only use if a new image that was not processed!
# return character segment snips

#TODO load in X_test, y_test

#TODO predict single character segment snip
# return class predicted


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
    X, y = load_data_to_predict(holdout_directory, sample_frac=0.5)

    # score = model.evaluate(X_test, y_test, verbose=0)
    # print('Test score:', score[0])
    # print(f'Test accuracy:, {score[1]}')  # this is the one we care about

    # predictions_labeled = zip_prediction_labels(predictions)





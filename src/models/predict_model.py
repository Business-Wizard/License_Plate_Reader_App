from src import features
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

#TODO import trained model as model

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
    '''load model with weights'''
    model = load_model("./models/model")

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print(f'Test accuracy:, {score[1]}')  # this is the one we care about




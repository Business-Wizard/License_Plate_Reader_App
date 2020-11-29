import src.features as features
from tensorflow import keras
import tensorflow as tf

#TODO import trained model as model

#TODO apply pipeline on new image
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

if __name__ == '__main__':
    '''single load of model + weights'''
    model = keras.models.load_model("../../models/")

    '''load architecture and weights separately'''
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    print("Completed loading model from disk")

    
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print(f'Test accuracy:, {score[1]}')  # this is the one we care about


    
    

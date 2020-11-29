from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential()  # model is a linear stack of layers (don't change)
    # note: the convolutional layers and dense layers require an activation function
    # see https://keras.io/activations/
    # and https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='same',
                     input_shape=input_shape))  #! first conv. layer KEEP
    model.add(Activation('relu'))  # Activation specification necessary for Conv2D and Dense layers
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='same',
                     input_shape=input_shape))  #! 2nd conv. layer KEEP
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='same',
                     input_shape=input_shape))  #! 3rd conv. layer  KEEP
    model.add(Activation('relu'))  # Activation specification necessary for Conv2D and Dense layers

    model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    
    model.add(Flatten())  #! necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(40))  #! (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))
    model.add(Dense(40))  #! (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    model.add(Dense(nb_classes))  #! 10 final nodes (one for each class)  KEEP
    model.add(Activation('softmax'))  #! softmax at end to pick between classes 0-9 KEEP

    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    #* suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    #* and KEEP metrics at 'accuracy'
    #? suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    #! important inputs to the model: don't changes the ones marked KEEP
    batch_size = 5000  # number of training samples used at a time to update the weights
    nb_classes = 10  #! number of output possibilities: [0 - 9] KEEP
    nb_epoch = 30    # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 28, 28   #! the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, 1)   #! 1 channel image input (grayscale) KEEP
    nb_filters = 15   # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (3, 3)  # convolutional kernel size, slides over image to learn features

    X_train, X_test, Y_train, Y_test = load_and_featurize_data()
    model = define_model(nb_filters, kernel_size, input_shape, pool_size)
    # during fit process watch train and test error simultaneously
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    
    # evaluate model on train data
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print(f'Test accuracy:, {score[1]}')  # this is the one we care about

    '''single save of model + weights'''
    model.save("../../models")


    '''save architecture and weights separately'''
    # serialize model to disk
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved weights to disk")
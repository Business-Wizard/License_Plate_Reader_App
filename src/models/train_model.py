import logging
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import layers, models

if TYPE_CHECKING:
    from pathlib import Path

    from keras.models import Model

from src.models.processhelpers import TRAIN_DIR, load_test_data

warnings.filterwarnings('ignore')
np.random.seed(101)  # set seed for reproducibility
sns.set_context('notebook')  # pyright: ignore [reportUnknownMemberType]

# ! to avoid some common specific hardware/environment errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logging.info(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logging.error(e)

# data_directory = os.path.join(os.getcwd(), "data")
# processed_directory = os.path.join(data_directory, "processed/2_recognition")
# holdout_directory = os.path.join(processed_directory, "holdout_set")
# train_directory = os.path.join(processed_directory, "train_set")


def define_model(num_filters, kernel_size, input_shape, pool_size, num_classes: int = 33):
    model = models.Sequential()  # model is a linear stack of layers (don't change)
    # note: convolutional and dense layers require an activation function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    model.add(
        layers.Conv2D(
            num_filters,
            (kernel_size[0], kernel_size[1]),
            padding='valid',
            input_shape=input_shape,
            activation='relu',
        ),
    )  # ! 1st conv. layer
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    model.add(layers.MaxPooling2D(pool_size=pool_size))

    model.add(layers.Flatten())
    # ! necessary to flatten before going into conventional dense layer
    logging.info('Model flattened out to ', model.output_shape)

    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # see https://keras.io/optimizers/#usage-of-optimizers
    # * KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # * and KEEP metrics at 'accuracy'
    # ? optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def save_model(model, destination: Path = './models/', filename: str = 'model'):
    logging.info('SAVING MODEL')
    tf.keras.models.save_model(
        model,
        destination / f'{filename}_full',
        include_optimizer=True,
        save_format='h5',
    )
    model.save_weights(destination / f'{filename}_weights', save_format='h5')


def visualize_history(_model: Model):
    title_names = ['Loss', 'Accuracy', 'Validation Loss', 'Validation Accuracy']
    _fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200, figsize=(11, 6))
    for key, name, ax in zip(model_history.history.keys(), title_names, ax.flatten(), strict=False):
        ax.plot(model_history.history[key], linewidth=3)
        ax.set_title(name)
        ax.set_xlabel('epoch')
        ax.set_ylabel(key)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, encoder = load_test_data(TRAIN_DIR, sample_frac=0.2)
    logging.info('DATA READY')

    model = define_model(
        num_filters=40,
        kernel_size=(4, 4),
        input_shape=(30, 30, 1),
        pool_size=(2, 2),
        num_classes=33,
    )
    logging.info(model.summary())

    model_history = model.fit(
        X_train,
        y_train,
        batch_size=20,
        epochs=10,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    score = model.evaluate(X_test, y_test, verbose=0)
    # predictions = model.predict(X_test)  #one-hot encoded
    # predictions = model.predict_classes(X_test).reshape((-1, 1)) #deprecated
    predictions_array = model.predict(X_test)
    predictions = np.argmax(predictions_array, axis=-1).reshape((-1, 1))
    y_test = np.argmax(y_test, axis=-1).reshape((-1, 1))
    logging.info(predictions[0:10])
    logging.info(predictions.shape)
    logging.info('####################################################')
    logging.info(y_test[0:10])
    logging.info(y_test.shape)
    # y_test = encoder.inverse_transform(predictions)
    # logging.info(y_test[0:10])
    # logging.info(y_test.shape)

    # logging.info(classification_report(y_test, predictions))
    logging.info(f'Test score: {score[0]}')
    logging.info(f'Test accuracy: {score[1]}')  # this is the one we care about

    visualize_history(model)
    # save_model(model)
    # logging.info(encoder.inverse_transform(predictions_array)[0:20])

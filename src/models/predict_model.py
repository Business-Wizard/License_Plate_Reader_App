import logging
import pathlib
from pathlib import Path

import tensorflow as tf
from keras import models
from keras.models import Model

from src.data.makedataset_recog import process_directory
from src.models.processhelpers import (
    HOLDOUT_DIR,
    encoder,
    load_single_image,
    load_test_data,
    load_unseen_data,
    zip_prediction_labels,
)

# ! to avoid some common hardware/environment errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logging.info(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError:
        logging.exception('Memory growth must be set before GPUs have been initialized')


def load_model_with_weights(model_name: str = 'model_full', directory: str = './models/') -> Model:
    model = models.load_model(directory + model_name)
    assert isinstance(model, Model)
    return model


def load_data_to_validate(source, sample_frac: float = 1.0):
    images_array, labels_array, encoder = load_test_data(
        HOLDOUT_DIR,
        sample_frac=0.01,
        validate=True,
    )
    logging.info(f'Images Shape: {images_array.shape}')
    logging.info(f'Labels Shape: {labels_array.shape}')
    return images_array, labels_array, encoder


def validate_model(model_name: str = 'model_full', holdout_data: str = HOLDOUT_DIR):
    logging.info('Validating Model')
    model: Model = load_model_with_weights(model_name=model_name)
    logging.info(model.summary())
    x, y, encoder = load_data_to_validate(HOLDOUT_DIR)
    score = model.evaluate(x, y, verbose='0')
    logging.info(f'Test score: {score[0]}')
    logging.info(f'Test accuracy: {score[1]}')
    predictions_array = model.predict(x)
    predictions_labeled = zip_prediction_labels(predictions_array, encoder)
    logging.info(f'Sample predictions: {predictions_labeled[:5]}')
    logging.info('Completed Validation', '\n')
    return predictions_labeled


def load_data_to_predict(source, sample_frac: float = 1.0):
    return load_unseen_data(source, sample_frac)


def predict_new_images(source, destination):
    model = load_model_with_weights('model_full')
    process_directory(UNPROCESSED_IMAGES_DIR, PREDICTION_IMAGES_DIR, size=-1)
    x = load_data_to_predict(PREDICTION_IMAGES_DIR)
    predictions_array = model.predict(x)
    predictions_labeled = zip_prediction_labels(predictions_array, encoder)
    return predictions_labeled, x, model


def predict_single_image(processed_image, model):
    if not model:
        model = load_model_with_weights('model_full')
    x = load_single_image(processed_image)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    x = x.astype('float32')
    x /= 255
    logging.info(type(x), x.shape)
    prediction = model.predict(x)
    return zip_prediction_labels(prediction, encoder)


if __name__ == '__main__':
    validate_model()

    # ! Directory with unprocessed images to be read
    UNPROCESSED_IMAGES_DIR: Path = pathlib.Path().cwd() / 'data/raw/'
    PREDICTION_IMAGES_DIR: Path = pathlib.Path().cwd() / 'data/processed/3_prediction/'
    predictions, X, model = predict_new_images(
        UNPROCESSED_IMAGES_DIR,
        PREDICTION_IMAGES_DIR,
    )
    logging.info(predictions)
    # ! uncomment to save predictions as a csv
    # np.savetxt("models/predictions.csv", predictions,
    #            delimiter=",", newline='\n', fmt='%s')

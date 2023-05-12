import logging
import pathlib
from pathlib import Path

import tensorflow as tf
from keras.models import Model, load_model

from src.data.makedataset_recog import process_directory
from src.models.processhelpers import (
    encoder,
    holdout_directory,
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
        logging.error('Memory growth must be set before GPUs have been initialized')


def load_model_with_weights(model_name: str = 'model_full', directory: str = './models/') -> Model:
    model = load_model(directory + model_name)
    assert isinstance(model, Model)
    return model


def load_data_to_validate(source, sample_frac: float = 1.0):
    images_array, labels_array, encoder = load_test_data(
        holdout_directory,
        sample_frac=0.01,
        validate=True,
    )
    logging.info(f'Images Shape: {images_array.shape}')
    logging.info(f'Labels Shape: {labels_array.shape}')
    return images_array, labels_array, encoder


def validate_model(model_name: str = 'model_full', holdout_data: str = holdout_directory):
    logging.info('Validating Model')
    model: Model = load_model_with_weights(model_name=model_name)
    logging.info(model.summary())
    X, y, encoder = load_data_to_validate(holdout_directory)
    score = model.evaluate(X, y, verbose='0')
    logging.info(f'Test score: {score[0]}')
    logging.info(f'Test accuracy: {score[1]}')
    predictions_array = model.predict(X)
    predictions_labeled = zip_prediction_labels(predictions_array, encoder)
    logging.info(f'Sample predictions: {predictions_labeled[:5]}')
    logging.info('Completed Validation', '\n')
    return predictions_labeled


def load_data_to_predict(source, sample_frac: float = 1.0):
    X = load_unseen_data(source, sample_frac)
    return X


def predict_new_images(source, destination):
    model = load_model_with_weights('model_full')
    process_directory(UNPROCESSED_IMAGES_DIR, PREDICTION_IMAGES_DIR, size=-1)
    X = load_data_to_predict(PREDICTION_IMAGES_DIR)
    predictions_array = model.predict(X)
    predictions_labeled = zip_prediction_labels(predictions_array, encoder)
    return predictions_labeled, X, model


def predict_single_image(processed_image, model):
    if not model:
        model = load_model_with_weights('model_full')
    X = load_single_image(processed_image)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = X.astype('float32')
    X /= 255
    logging.info(type(X), X.shape)
    prediction = model.predict(X)
    prediction_labeled = zip_prediction_labels(prediction, encoder)
    return prediction_labeled


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

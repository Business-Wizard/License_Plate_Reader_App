import tensorflow as tf
from tensorflow.keras.models import load_model

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
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def load_model_with_weights(model_name: str = "model_full",
                            directory: str = "./models/"):
    model = load_model(directory + model_name)
    return model


def load_data_to_validate(source, sample_frac: float = 1.0):
    images_array, labels_array, encoder =\
         load_test_data(holdout_directory,
                        sample_frac=0.01, validate=True)
    print(f"Images Shape: {images_array.shape}")
    print(f"Labels Shape: {labels_array.shape}")
    return images_array, labels_array, encoder


def validate_model(model: str = "model_full",
                   holdout_data: str = holdout_directory):
    print("Validating Model")
    model = load_model_with_weights(model)
    print(model.summary())
    X, y, encoder = load_data_to_validate(holdout_directory)
    score = model.evaluate(X, y, verbose=0)
    print(f"Test score: {score[0]}")
    print(f"Test accuracy: {score[1]}")
    predictions_array = model.predict(X)
    # predictions = np.argmax(predictions_array, axis=-1).reshape(-1, 1)
    predictions_labeled = zip_prediction_labels(predictions_array, encoder)
    print(f"Sample predictions: {predictions_labeled[:5]}")
    print("Completed Validation", '\n')
    return predictions_labeled


def load_data_to_predict(source, sample_frac: float = 1.0):
    X = load_unseen_data(source, sample_frac)
    return X


def predict_new_images(source, destination):
    model = load_model_with_weights("model_full")
    process_directory(unprocessed_images_directory,
                      prediction_images_directory,
                      size=-1)
    X = load_data_to_predict(prediction_images_directory)
    predictions_array = model.predict(X)
    # predictions = np.argmax(predictions_array, axis=-1).reshape(-1, 1)
    predictions_labeled = zip_prediction_labels(predictions_array, encoder)
    return predictions_labeled, X, model


def predict_single_image(processed_image, model):
    if not model:
        model = load_model_with_weights("model_full")
    X = load_single_image(processed_image)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = X.astype('float32')
    X /= 255
    print(type(X), X.shape)
    prediction = model.predict(X)
    prediction_labeled = zip_prediction_labels(prediction, encoder)
    return prediction_labeled


if __name__ == '__main__':
    validate_model()

    # ! Directory with unprocessed images to be read
    unprocessed_images_directory = "./data/raw/"
    prediction_images_directory = "./data/processed/3_prediction/"
    predictions, X, model = predict_new_images(unprocessed_images_directory,
                                               prediction_images_directory)
    print(predictions)
    # ! uncomment to save predictions as a csv
    # np.savetxt("models/predictions.csv", predictions,
    #            delimiter=",", newline='\n', fmt='%s')

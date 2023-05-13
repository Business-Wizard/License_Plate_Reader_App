import logging
import pathlib
import secrets
from pathlib import Path
from typing import Final

import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
from keras import models
from keras.models import Model

# import flask_session
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Response

from src.data.imageprocess import pipeline_single
from src.models.predict_model import predict_single_image

APP_ROOT_DIR: Final[Path] = pathlib.Path().cwd()
UPLOAD_FOLDER: Final[Path] = APP_ROOT_DIR / 'web_app/static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
SESSION_TYPE = 'filesystem'
SESSION_FILE_DIR: Final[Path] = APP_ROOT_DIR / 'flask_session'
SESSION_FILE_THRESHOLD = 5

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048
secret = secrets.token_urlsafe(32)
app.secret_key = secret
# flask_session.Session(app)


def is_supported_file_type(filename) -> bool:
    has_extension: bool = '.' in filename
    extension_allowed: bool = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return has_extension and extension_allowed


def predict_license_plate(model, img_path: Path) -> np.ndarray:
    logging.info(img_path)
    image = pipeline_single(filepath=img_path)
    return predict_single_image(image, model)


# home page
@app.route('/')
def upload_form() -> str:
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image() -> Response:
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if not file.filename:
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and is_supported_file_type(file.filename):
        filename = secure_filename(file.filename)
        logging.info(f'upload_image filename: {filename}')
        file.save(app.config['UPLOAD_FOLDER'] / filename)
        session['raw_image'] = filename

        preds: np.ndarray = predict_license_plate(model, app.config['UPLOAD_FOLDER'] / filename)
        preds = str(preds[0]) if len(preds) == 1 else preds
        logging.info(type(preds), preds)
        preds_name: str = f'predicted_{filename}'
        session['prediction'] = preds
        logging.info(
            '#######################',
            filename,
            preds_name,
            preds,
        )
        # ! os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('display_image'))

    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/')
def display_image() -> str:
    prediction = session.get('prediction')
    raw_image = session['raw_image']
    logging.info(raw_image)
    # raw_image = SavedModel file does not exist at: models/model_full/{saved_model.pbtxt|saved_model.pb}(app.config['UPLOAD_FOLDER'], original_image)
    # return redirect(url_for('static', filename='uploads/' + raw_image), code=301)
    return render_template('results.html', raw_image=raw_image, prediction=prediction)
    # ! get image displayed!!
    # return send_from_directory(app.config['UPLOAD_FOLDER'], original_image), prediction


if __name__ == '__main__':
    model_location = APP_ROOT_DIR / 'models/model_full'
    model: Model = models.load_model(model_location)
    logging.info('Model loaded. Start serving...')

    # http_server = WSGIServer(('0.0.0.0', 31000), app)
    # http_server.serve_forever()

    app.run(host='0.0.0.0', port=31000, threaded=True, debug=False)  # noqa: S104

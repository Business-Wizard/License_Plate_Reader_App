import os
import pickle
from src.models.processhelpers import load_single_image, load_unseen_data
import sys
import numpy as np
import pandas as pd
from src.data.imageprocess import pipeline_single, save_image
from src.models.predict_model import predict_single_image

from flask import (Flask, redirect, render_template, request, url_for,
                   flash, send_from_directory, session)
# import flask_session
from werkzeug.utils import secure_filename
import secrets

from tensorflow.keras.models import load_model


UPLOAD_FOLDER = os.getcwd() + '/web_app/static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
SESSION_TYPE = 'filesystem'
SESSION_FILE_DIR = os.getcwd() + "flask_session"
SESSION_FILE_THRESHOLD = 5

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048
secret = secrets.token_urlsafe(32)
app.secret_key = secret
# flask_session.Session(app)


def allowed_file(filename):
    has_extension = '.' in filename
    extension_allowed = filename.rsplit('.', 1)[1]\
        .lower() in ALLOWED_EXTENSIONS
    return has_extension and extension_allowed


def predict_license_plate(model, img_path):
    print(type(img_path))
    print(img_path)
    image = pipeline_single(img_path)  # blurred image
    # image = load_single_image(image)
    prediction = predict_single_image(image, model)
    return prediction


# home page
@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(f'upload_image filename: {filename}')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        session['raw_image'] = os.path.join(
            app.config['UPLOAD_FOLDER'], filename)

        preds =\
            predict_license_plate(model,
                                  os.path.join(app.config['UPLOAD_FOLDER'],
                                               filename))
        preds = str(preds[0]) if len(preds) == 1 else preds
        print(type(preds), preds)
        preds_name = 'predicted_' + filename
        session['prediction'] = preds
        print("#######################",
              filename, preds_name, preds,
              type(filename), type(preds_name))
        # ! os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('display_image'))

    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<prediction>')
def display_image():
    prediction = session.get('prediction')
    raw_image = session['raw_image']
    # raw_image = os.path.join(app.config['UPLOAD_FOLDER'], original_image)
    # return redirect(url_for('static', filename='uploads/' + raw_image), code=301)
    print(raw_image)
    return render_template("results.html",
                           raw_image=raw_image,
                           prediction=prediction)
                           # ! get image displayed!!
    # return send_from_directory(app.config['UPLOAD_FOLDER'], original_image), prediction


if __name__ == '__main__':
    model = load_model('models/model_full')
    print('Model loaded. Start serving...')

    # http_server = WSGIServer(('0.0.0.0', 31000), app)
    # http_server.serve_forever()

    app.run(host='0.0.0.0', port=31000, threaded=True, debug=False)


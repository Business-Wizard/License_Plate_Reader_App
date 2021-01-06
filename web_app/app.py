import os
import pickle
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
SESSION_FILE_THRESHOLD = 10

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
    processed_image = pipeline_single(img_path)
    prediction = predict_single_image(processed_image)
    return prediction


# home page
@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        # return render_template('index.html')
        return redirect(url_for('upload_image'))
    # TODO render button to route to upload page


@app.route('/upload/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('upload.html')

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        file_names = []
        for file in files:
            # if file.filename == '':
            #     flash('No image selected for uploading')
            #     return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_names.append(filename)
                session['raw_image'] = filename
                print(f'upload_image filename: {filename}')
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # ! os.remove(os.path.join(app.config['UPLOAD_FOLDER'],
                                    #    filename))
            else:
                flash('Allowed image types are -> png, jpg, jpeg, gif')
                return redirect(request.url)

            return render_template('upload.html', filenames=file_names)


            '''
            preds =\
                predict_license_plate(model,
                                      os.path.join(app.config['UPLOAD_FOLDER'],
                                                   filename))
            preds_name = 'predicted_' + filename
            save_image(os.path.join(app.config['UPLOAD_FOLDER'],
                                    preds_name),
                       preds)
            print("#######################",
                  filename, preds_name,
                  type(filename), type(preds_name))
            return redirect(url_for('display_image', raw_image=filename,
                                    processed=preds_name))
            '''


@app.route('/display/<raw_image>&<processed>/')
def display_image(raw_image, processed):
    return send_from_directory(app.config['UPLOAD_FOLDER'], processed),\
           send_from_directory(app.config['UPLOAD_FOLDER'], raw_image)
    # ! display both images


'''
# * may not be necessary (do upload & predict in /upload)
@app.route('/predict/', methods=['GET'])
def get_prediction():
    pass
    return preds
    else:
        return "No Image Received"
'''



if __name__ == '__main__':
    model = load_model('models/model_full')
    print('Model loaded. Start serving...')

    # http_server = WSGIServer(('0.0.0.0',31000), app)
    # http_server.serve_forever()

    app.run(host='0.0.0.0', port=31000, threaded=True, debug=False)



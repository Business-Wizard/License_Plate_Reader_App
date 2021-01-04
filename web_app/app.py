import os
import pickle
import sys
import numpy as np
import pandas as pd
from src.data.imageprocess import pipeline_single

from flask import Flask, redirect, render_template, request, url_for, flash
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model

def predict_license_plate(model, img_path):
    image = pipeline_single(img_path)

app = Flask(__name__)

# home page
@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    # TODO render button to route to upload page

@app.route('/upload', methods=['GET'])
def upload():
    # TODO Flask upload feature to get image
    pass


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            file.save(filename)
            preds = predict_license_plate(model, filename)
            os.remove(filename)
        return preds 




if __name__ == '__main__':
    model = load_model('models/model_full')
    print('Model loaded. Start serving...')

    # http_server = WSGIServer(('0.0.0.0',31000), app)
    # http_server.serve_forever()

    app.run(host='0.0.0.0', port=31000, threaded=True, debug=True)



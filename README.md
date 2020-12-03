# License Plate Recognition

## Using Convolusional Neural Networks to recognize license plates

-----------------------------------------

1. background:
   1. WHY? Artificial Intelligence is quickly part of every product or service we, as consumers, interact with.  AI is also making entirely new services and industries possible.
   2. Have you ever ordered food for takeout during Covid-19?  Even though you've given your vehicle information, you still have to call the restaurant or blink your lights to let them know you're ready to pick-up.
   3. Now imagine a second scenario - you pull up in the parking lot, your license plate is recognized by a camera, the staff are notified you're ready to pick up your order, and either the staff or app can offer you your favorite sides, sauces, drinks, or even discount coupons based on your previous orders.
   4. Small businesses and franchise chains can add new features of acknowledging customers as well as gaining insight into customer loyalty.  Such insights can be used for individually tailored marketing, much like how your web browser provides a litany of information about your viewing and purchasing habits such as your favorite brands, 
   5. Currently there is a huge cost for license plate reading solutions - small businesses can't take advantage
2. WHAT? Detect and Recognize car license plates

3. HOW? use CNN (convolutional neural network) to read license plates

4. data description:
   1. source *footnote* ####################################3
   2. *image of license plate* ######################################
   3. (100000, 218, 1125, 3)
       1. count 100k
       2. dimensions

5. image processing:
   1. License Plate images
      1. license plates used 33 characters - 10 numbers and 23 letters (no Q, W, or X)
      2. size of (218, 1125)
      3. The images are originally in color - represented by 3 color channels, Red Green, Blue (RGB) *image of grayscale*  ########################################
      4. thresholded  #####################################
      5. One of the first designs used a morphological operator, dilation, to smooth the thresholded images, but this also led to the system misidentifying "LA" as a single letter.
      6. so I opted to instead use an erosion, to increase the distance between letters, followed by a gaussian blur to smooth the edges.
      7. erosion  ##################################
      8. blurred  ##################################
      9.  
   2. contour detection
      1. 

6. model architecture:
   1. While testing multiple architecture designs, I realized the models would very easily get to 100% accuracy for this generated dataset.  The final model for this phase of the project was selected for its simplicity, fast training time, and small size (< 3 MB) that could be deployed on mobile devices and web browsers.
   2. design
      1. image inputs (30, 30, 1)
      2. convolution kernel (4, 4) x 40 filters, no padding on images
         1. output (27, 27, 40)
      3. max pooling (sub-sampling)
         1. output (13, 13, 40)
      4. max pooling (sub-sampling)
         1. output (6, 6, 40)
      5. flatten
         1. (1440)
      6. dense neural network 20 relu
      7. dense neural network 33 softmax
   3. optimizer adam, loss categorical cross entropy
   4. total parameters: 30,193

7. model performance / size:
   1. the CNN was trained on grayscale images of size (30, 30)
   2. 20 images were batched and sent through the CNN for classification during a single epoch.
   3. After each epoch, the CNN would update each of the different feature maps' weights in order to improve its classification performance.
   4. This process repeated for 10 epochs total, as the model would quickly reach feature weights for > 90% accuracy within 4 epochs.

8. technology stack:
   1. python
   2. keras
   3. tensorflow
   4. opencv
   5. numpy
   6. scikit-learn
   7. matplotlib
   8. seaborn
   9. flake8

9.  installation:
    1.  how to git clone
    2.  python scripts for setup.py
    3.  where to put license plate images


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    |   └── raw            <- New unprocessed images for reading.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

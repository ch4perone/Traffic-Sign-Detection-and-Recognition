import argparse
import utils
import numpy as np
import readTrafficSigns
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.models import Sequential
from keras.layers import Activation, Dense, Convolution2D, Dropout
from keras.layers import Flatten, MaxPooling2D, BatchNormalization
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from IPython.display import SVG
import matplotlib.pyplot as plt
from keras.models import load_model

# %matplotlib inline

parser = argparse.ArgumentParser(description='Supervised Street Sign Assignment (s4a): Machine Learning method for '
                                             'learning and recognition of street signs')
parser.add_argument('-T', '--test_data', type=str,
                    help='test data [directory]')
parser.add_argument('-v', '--verbose', type=int,
                    help='verbose', default=0)
parser.add_argument('-M', '--model', type=str,
                    help='input model')

args = parser.parse_args()


def vprint(string):
    if args.verbose == 1:
        print(string)


# returns a compiled model
model = load_model(args.model)
vprint("Loaded model from: " + args.model)

images, labels = readTrafficSigns.readTestTrafficSigns(args.test_data)

labels = [int(l) for l in labels]
labels = utils.preprocess_labels(labels, size=max(labels))
images = np.array(images)

vprint('Testing Model')
pred = model.predict(images)

vprint("Accuracy: {}".format(accuracy_score(pred > 0.5, labels)))
# vprint("ROC AUC: {}".format(roc_auc_score(pred > 0.5, y_test)))
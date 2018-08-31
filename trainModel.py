import argparse
import numpy as np
import readTrafficSigns
import random
import utils
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
#%matplotlib inline

# usage: python3.6 trainModel.py -D ./data/GTSRB/Final_Training/Images -v 1 -O modelname
parser = argparse.ArgumentParser(description='Supervised Street Sign Assignment (s4a): Machine Learning method for '
                                             'learning and recognition of street signs')
parser.add_argument('-D', '--data', type=str,
                    help='input data [Image directory]')
parser.add_argument('-v', '--verbose', type=int,
                    help='verbose', default=0)
parser.add_argument('-O', '--out', type=str,
                    help='output model name [without file ending]', default='model')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()


def vprint(string):
    if args.verbose:
        print(string)


vprint('begin loading images from file')
images, labels = readTrafficSigns.readTrafficSigns(args.data)


vprint('\tdone\nplot exemplary images')
fig = plt.figure(figsize=(20, 5))
for i in range(30):
    ax = plt.subplot(3, 10, i + 1)
    rand_sample = random.randrange(0, len(images))
    plt.imshow(images[rand_sample], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

vprint('preprocess input data')

labels = [int(l) for l in labels]
labels = utils.preprocess_labels(labels, size=max(labels))
images = np.array(images)
#df = pd.DataFrame(images)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
vprint('\tdone')

# do assert shape here
vprint('begin setting up model')
model = Sequential()
#model.add(Dropout(input_shape=images[0].shape, rate=0.1))
model.add(
    Convolution2D(input_shape=images[0].shape, name='Convolution', strides=(1, 1), data_format='channels_last', filters=10,
                  kernel_size=(10, 10)))
model.add(BatchNormalization())
model.add(Activation(activation='relu', name='activation'))
model.add(MaxPooling2D(pool_size=(5, 5), name='maxPooling', data_format='channels_last'))
#model.add(Dropout(0.1))
model.add(Convolution2D(name='Convolution2', strides=(1, 1), data_format='channels_last', filters=10, kernel_size=(10, 10)))
model.add(BatchNormalization())
model.add(Activation(activation='relu', name='activation2'))
model.add(MaxPooling2D(pool_size=(5, 5), name='maxPooling2', data_format='channels_last'))
model.add(Flatten(name='flatten'))
#model.add(Dropout(0.1))
model.add(Dense(units=640, name='dense'))
model.add(Dense(units=labels.shape[1], activation='softmax'))

# sgd = SGD(lr=0.1, decay=1e-2, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
    )

vprint('\tdone')


vprint('training model')
EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='min')

history = model.fit(x=X_train, y=y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[EarlyStop], shuffle=False)
vprint('training done')


# pred = model.predict(X_test)

# vprint("Accuracy: {}".format(accuracy_score(pred > 0.5, y_test)))
# vprint("ROC AUC: {}".format(roc_auc_score(pred > 0.5, y_test)))

fig = plt.figure(figsize=(14,14))
# make it two rows with one plot each
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


model.save(args.out + '.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
vprint("Saved model to disk")




fig = plt.figure(figsize=(15, 3))
for i in range(10):
    ax = plt.subplot(2, 8, i+1)
    # the weight matrices are in the format
    # (nrow, ncol, 1, num_of_filters)
    weights_i = model.layers[0].get_weights()[0][:,:,0, i]
    plt.imshow(weights_i, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

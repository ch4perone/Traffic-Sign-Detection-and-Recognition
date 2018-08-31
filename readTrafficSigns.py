# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
from PIL import Image
import numpy as np
from skimage.transform import resize

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        shape = (100, 100)
        for row in gtReader:
            #images.append(resize(plt.imread(prefix + row[0]), output_shape=shape, anti_aliasing=True, mode='reflect', order=1)) # the 1th column is the filename
            img = Image.open(prefix + row[0])
            img = img.resize(shape, Image.ANTIALIAS)
            images.append(np.array(img))
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels


def readTestTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    prefix = rootpath + '/' # subdirectory for class
    gtFile = open(prefix + 'GT-final_test.test.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header
    # loop over all images in current annotations file
    shape = (100, 100)
    for row in gtReader:
        #images.append(resize(plt.imread(prefix + row[0]), output_shape=shape, anti_aliasing=True, mode='reflect', order=1)) # the 1th column is the filename
        img = Image.open(prefix + row[0])
        img = img.resize(shape, Image.ANTIALIAS)
        images.append(np.array(img))
        labels.append(row[7]) # the 8th column is the label
    gtFile.close()
    return images, labels


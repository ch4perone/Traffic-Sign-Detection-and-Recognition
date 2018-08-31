import argparse
import utils
import numpy as np
from sklearn.svm import SVC
import pickle
import timeit

parser = argparse.ArgumentParser(description='Supervised Street Sign Assignment (s4a): Machine Learning method for '
                                             'learning and recognition of street signs')
parser.add_argument('-v', '--verbose', type=int,
                    help='verbose', default=0)
parser.add_argument('-T', '--trainSVM', action='store_true', help='tag if model should train')
parser.add_argument('-L', '--loadModel', type=str, default=None,
                    help='load preexisting SVM')
parser.add_argument('-I', '--image', type=str, default=None,
                    help='image to classify')


args = parser.parse_args()


if args.trainSVM:
    print('Load feature data from images')
    print('Load positive set: ')
    X = utils.extractFeatureMatrixFromPicture('./data/positiveRed.jpg', subset=1)
    y = np.ones(X.shape[0])


    print('Load negative set: ')
    X_neg = utils.extractFeatureMatrixFromPicture('./data/negative.jpg', subset=1)
    X = np.concatenate((X, X_neg), axis=0)
    y = np.append(y, np.zeros(X_neg.shape[0]))

    print(X.shape)
    print(y.shape)


    print('\tdone')
    print('Setup classifier (SVM)')
    clf = SVC(kernel='linear', probability=True)
    print('Train classifier')
    start = timeit.default_timer()

    clf.fit(X, y)
    print('\tdone')
    stop = timeit.default_timer()
    print('train time [sec]: \n\t')
    print(stop - start)

    print('classify test image')
    #utils.extractRegion(args.image, clf, plot=True)
    filename = 'svmTest.sav'
    pickle.dump(clf, open(filename, 'wb'))

#Your statements here



else:
    print('Load existing model (SVM)')
    clf = pickle.load(open(args.loadModel, 'rb'))

    print('classify test image')
    utils.extractRegion(args.image, clf, threshold=0.8, plot=True)

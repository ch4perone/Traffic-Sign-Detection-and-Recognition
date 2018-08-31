import numpy as np
from PIL import Image
import numpy as np
import colorsys
import sys
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KernelDensity


def vectorize_label(number, size):
    return [1 if i == number else 0 for i in range(size)]


def preprocess_labels(categoricals, size):
    return np.array([vectorize_label(y, size) for y in categoricals])


def extractFeatureMatrixFromPicture(picturePath, subset=1):
    im = Image.open(picturePath)
    pix = im.load()
    width, height = im.size  # Get the width and height of the image for iterating over
    pixnum = width * height

    featureMatrix = np.array([[0, 0, 0, 0, 0, 0]])
    pixc = 0
    for i in range(0, width):
        if pixc > subset * pixnum:
            break
        for j in range(0, height):
            pixc += 1
            if pix[i, j][0] > 252 and pix[i, j][1] > 252 and pix[i, j][2] > 252: continue
            hsv = colorsys.rgb_to_hsv(pix[i, j][0], pix[i, j][1], pix[i, j][2])
            ma = np.array([pix[i, j][0], pix[i, j][1], pix[i, j][2], hsv[0], hsv[1], hsv[2]])
            featureMatrix = np.concatenate((featureMatrix, [ma]))
            sys.stdout.write("\r" + str(pixc * 100 / float(pixnum))[:4] + '%')
    sys.stdout.write('\n')
    featureMatrix = np.delete(featureMatrix, 0, axis=0)
    return featureMatrix


# https://gist.github.com/daleroberts/7a13afed55f3e2388865b0ec94cd80d2
def plotKDE(x, y, size):
    xy = np.vstack([x, y])

    #d = xy.shape[0]
    #n = xy.shape[1]
    #bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))  # silverman
    for bw in [1,5,10,20,50]:
        # bw = n**(-1./(d+4)) # scott
        print('bw: {}'.format(bw))

        kde = KernelDensity(bandwidth=bw, metric='euclidean',
                            kernel='gaussian', algorithm='ball_tree')
        kde.fit(xy.T)

        xmin = 0
        xmax = size[0]
        ymin = 0
        ymax = size[1]

        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        Z = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)

        plt.imshow(np.flip(np.rot90(Z, k = 3), axis=1), cmap=plt.cm.viridis,
                   extent=[0, size[0], 0, size[1]], origin='upper')

        # plt.scatter(x, y, c='k', s=5, edgecolor='', ) rotate somehow
        plt.show()





def extractRegion(picturePath, clf, threshold, plot=False):
    im = Image.open(picturePath)
    pix = im.load()
    width, height = im.size  # Get the width and height of the image for iterating over
    pixnum = width * height
    pixc = 0
    importantPix = []
    xs, ys = [], []
    for i in range(0, width):
        for j in range(0, height):
            pixc += 1
            sys.stdout.write("\r" + str(pixc * 100 / float(pixnum))[:4] + '%')
            hsv = colorsys.rgb_to_hsv(pix[i, j][0], pix[i, j][1], pix[i, j][2])
            ma = np.array([pix[i, j][0], pix[i, j][1], pix[i, j][2], hsv[0], hsv[1], hsv[2]])
            # featureMatrix = np.concatenate((featureMatrix,[ma]))
            # sys.stdout.write("\r" + str(countc / 160000.0))
            #lab = clf.predict([ma])
            pred = clf.predict_proba([ma])
            #print(pred, lab)
            if pred[0][1] > threshold:
                importantPix.append((i, j))
                xs.append(i)
                ys.append(j)

    sys.stdout.write('\n')

    if plot:
        clfMatrix = np.zeros((height, width, 3))
        for p in importantPix:
            clfMatrix[p[1], p[0]] = (255, 255, 255)

        # img = Image.fromarray(clfMatrix, 'RGB')
        plt.imshow(clfMatrix, origin='upper')
        plt.show()

        plotKDE(xs, ys, size=(width, height))

    return importantPix


def testClf(picturePath, clf):
    im = Image.open(picturePath)
    pix = im.load()

    X = np.array(pix)
    lab = clf.predict(X)

    return lab

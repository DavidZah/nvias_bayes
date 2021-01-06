import os

import cv2
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np


dataset = datasets.load_iris()
model = GaussianNB()
model.fit(dataset.data, dataset.target)

expected = dataset.target
predictetd = model.predict(dataset.data)


img_one = cv2.imread('numbers\\one.jpg', cv2.IMREAD_GRAYSCALE)
img_two = cv2.imread('numbers\\two.jpg', cv2.IMREAD_GRAYSCALE)

# define a threshold, 128 is the middle of black and white in grey scale
thresh = 128

# threshold the image
img_binary_one = np.array(cv2.threshold(img_one, thresh, 255, cv2.THRESH_BINARY)[1]).ravel()
img_binary_two = np.array(cv2.threshold(img_two, thresh, 255, cv2.THRESH_BINARY)[1]).ravel()

img_two_test = cv2.imread('numbers\\two.jpg', cv2.IMREAD_GRAYSCALE)
img_binary_two_test = np.array(cv2.threshold(img_two_test, thresh, 255, cv2.THRESH_BINARY)[1]).ravel()


model.fit([img_binary_one,img_binary_two], ['jedna','dva'])
print(model.predict(img_binary_two_test.reshape(1, -1)))

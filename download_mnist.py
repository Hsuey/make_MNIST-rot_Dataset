datasetType = "FashionMNIST" #  FashionMNIST or MNIST
saveDir = ""
# datasetType = 'w'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision
from torch.utils import data
from torchvision import transforms
from imageio import imsave

"""
From:
https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390
"""

import gzip
import os
import sys
import time

from six.moves import urllib
from six.moves import xrange    # pylint: disable=redefined-builtin
# from scipy.misc import imsave
# import tensorflow as tf
import numpy as np
import csv

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'raw_data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

# def maybe_download(filename):
#     """Download the data from Yann's website, unless it's already here."""
#     if not tf.gfile.Exists(WORK_DIRECTORY):
#         tf.gfile.MakeDirs(WORK_DIRECTORY)
#     filepath = os.path.join(WORK_DIRECTORY, filename)
#     if not tf.gfile.Exists(filepath):
#         filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
#         with tf.gfile.GFile(filepath) as f:
#             size = f.size()
#         print('Successfully downloaded', filename, size, 'bytes.')
#     return filepath
def download_mnist(data_location, datasetType="MNIST") -> "None":
    """Download the MNIST dataset and then load it into memory.

    Defined in :numref:`sec_mnist`"""

    if not os.path.exists(data_location):
        os.makedirs(data_location)
    # FashionMNIST
    if datasetType == "MNIST":
        mnist_train = torchvision.datasets.MNIST(
            root=data_location, train=True, download=True)
        mnist_test = torchvision.datasets.MNIST(
            root=data_location, train=False, download=True)
        path = data_location + "/MNIST/raw/"
        print(f"MNIST datasets locate in '{path}'")
    elif datasetType == "FashionMNIST":
        mnist_train = torchvision.datasets.FashionMNIST(
            root=data_location, train=True, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root=data_location, train=False, download=True)
        path = data_location + "/FashionMNIST/raw/"
        print(f"FashionMNIST datasets locate in '{path}'")
    else:
        raise ValueError("datasetType is error!!!")
    # for i in os.listdir(data_location + "/MNIST/raw/"):
    #     if i[-3:] != '.gz':
    #         print(path + i)
    return mnist_train, mnist_test


if __name__ == "__main__":
    data_location = "./data/"
    mnistTrain, mnistTest = download_mnist(data_location, datasetType=datasetType)

    train_data = np.array(mnistTrain.data.reshape(*(list(mnistTrain.data.shape)), 1))
    test_data = np.array(mnistTest.data.reshape(*(list(mnistTest.data.shape)), 1))
    train_labels = np.array(mnistTrain.targets)
    test_labels = np.array(mnistTest.targets)
    print(test_labels)
#     这里是原来在project中判断是否在正确的路径，这里由于是用jupyter notebook编写的代码，所以无需判断
#     if 'mnist' not in os.getcwd():
#         print('Path Error!')
#         raise ValueError
    if not os.path.isdir("./train-images"):
        os.makedirs("./train-images")
    if not os.path.isdir("./test-images"):
        os.makedirs("./test-images")

    # process train data
    with open("./train-labels.csv", 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        for i in range(len(train_data)):
            imsave("./train-images/" + str(i) + ".jpg", train_data[i][:, :, 0])
            writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])
        print("train-labels.csv OK !!!")
            
    # repeat for test data
    with open("./test-labels.csv", 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        for i in range(len(test_data)):
            imsave("./test-images/" + str(i) + ".jpg", test_data[i][:, :, 0])
            writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])
        print('test-labels.csv OK !!!')
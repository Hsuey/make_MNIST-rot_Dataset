datasetType = "FashionMNIST" #  FashionMNIST or MNIST
saveDir = ""
# datasetType = 'w'

import numpy as np
import os
from mnist import random_rotation, loadMnist


def makeMnistRot():
    """
    Make MNIST-rot from MNIST
    Select all training and test samples from MNIST and select 10000 for train,
    2000 for val and 50000 for test. Apply a random rotation to each image.

    Store in numpy file for fast reading

    """
    np.random.seed(0)
    
    #Get all samples
    all_samples = loadMnist('train') + loadMnist('test')

    #Empty arrays
    train_data = np.zeros([28,28,10000])
    train_label = np.zeros([10000])
    val_data = np.zeros([28,28,2000])
    val_label = np.zeros([2000])
    test_data = np.zeros([28,28,50000])
    test_label = np.zeros([50000])

    #new Empty arrays
    new_train_data = np.zeros([10000, 28,28])
    new_val_data = np.zeros([2000, 28,28])
    new_test_data = np.zeros([50000, 28,28])

    i = 0
    for j in range(10000):
        sample =all_samples[i]
        train_data[:, :, j] =  random_rotation(sample[0])
        new_train_data[j, :, :] =  train_data[:, :, j]
        train_label[j] = sample[1]
        i += 1

    for j in range(2000):
        sample = all_samples[i]
        val_data[:, :, j] = random_rotation(sample[0])
        new_val_data[j, :, :] = val_data[:, :, j]
        val_label[j] = sample[1]
        i += 1

    for j in range(50000):
        sample = all_samples[i]
        test_data[:, :, j] = random_rotation(sample[0])
        new_test_data[j, :, :] = test_data[:, :, j]
        test_label[j] = sample[1]
        i += 1
    if datasetType == "FashionMNIST":
        saveDir = "fashionmnist_rotation_new"
    else:
        saveDir = "mnist_rotation_new"
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    new_train = np.hstack((new_train_data.reshape(10000, -1), train_label.reshape(10000, 1)))
    new_val = np.hstack((new_val_data.reshape(2000, -1), val_label.reshape(2000, 1)))
    new_train_val = np.vstack((new_train, new_val))
    new_test = np.hstack((new_test_data.reshape(50000, -1), test_label.reshape(50000, 1)))
    print(f"trainslation finished: train_val_shape={new_train_val.shape}, test_shape={new_test.shape}")
    new_train_file_name = saveDir + '/mnist_all_rotation_normalized_float_train_valid.amat'
    new_test_file_name = saveDir + '/mnist_all_rotation_normalized_float_test.amat'
    np.savetxt(new_train_file_name, new_train_val, fmt="%.6f")
    print(f"save the {new_train_file_name} finished...\nlocate in {os.getcwd() + '/' + new_train_file_name}")
    np.savetxt(new_test_file_name, new_test, fmt="%.6f")
    print(f"save the {new_test_file_name} finished...\nlocate in {os.getcwd() + '/' + new_test_file_name}")

    try:
        os.mkdir('mnist_rot/')
    except:
        None

    np.save('mnist_rot/train_data',train_data)
    np.save('mnist_rot/train_label', train_label)
    np.save('mnist_rot/val_data', val_data)
    np.save('mnist_rot/val_label', val_label)
    np.save('mnist_rot/test_data', test_data)
    np.save('mnist_rot/test_label', test_label)

if __name__ == '__main__':
    makeMnistRot()
#     # test new_MNIST_rotation
#     train_file_name = 'mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat'
#     test_file_name = 'mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat'
#     import numpy as np
#     with open(train_file_name) as data_file:
#         data = np.loadtxt(data_file)
#     data.shape
#     plt.imshow(data[0][:-1].reshape(28, -1))
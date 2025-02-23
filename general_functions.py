# Load libraries
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


# Create the function to load the h5py train and test datasets & the classes
def load_h5py_dataset(train_path,test_path):

    # We read the train path 
    train_dataset = h5py.File(train_path, "r")
    # We split/create the X and Y partm and transform to array
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    # We read the test path + we split/create the X and Y partm and transform to array
    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    # We get the list of classes
    classes = np.array(test_dataset["list_classes"][:])
    
    # We reshape the Y train and test datasets
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Create the function to one-hot encode a vector (transform a vector to as many 0-1 vectors as nb_classes)
def convert_to_one_hot(vector, nb_classes):
    vector = np.eye(nb_classes)[vector.reshape(-1)].T
    return vector


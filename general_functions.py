# Load libraries
import math
import numpy as np
# Used to work with h5 files (designed to store and organize large amounts of data)
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf


# Create the function to load the train and test datasets (h5 files) & the classes
def load_h5_dataset(train_path, test_path):
    '''
    Load the h5 file for train and test datasets

    Inputs:
    train_path -- path of the train file (string)
    test_path -- path of the test file (string)

    Returns: 
    train_set_x_orig -- X part of the train set (array)
    train_set_y_orig --  Y part of the train set (array)
    test_set_x_orig --  X part of the test set (array)
    test_set_y_orig --  Y part of the test set (array)
    classes -- unique classes of the train set (list)
    '''
    # Read the train path 
    train_dataset = h5py.File(train_path, "r")
    # Split/create the X and Y parts and transform to array
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    # Read the test path + split/create the X and Y parts and transform to array
    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    # Get the list of classes
    classes = np.array(train_dataset["list_classes"][:])
    
    # Reshape the Y train and test datasets (not good practise to use rank 1 array)
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Create a function to transform an array to a one-hot array with n classes
def convert_to_one_hot(array, n):
    """
    Convert an array to a one-hot array with n classes
    
    Arguments:
    array -- array with max n different values
    n -- number of classes (int)
    
    Returns:
    encoded_array -- one-hot version of the input array
    """
    # np.eye(n): creates an identity matrix (so filled with 0, and 1 on the diagonal) with dimension (n,n)
    # [array.reshape(-1)]: flattens the array into 1D vector
    # We keep the xth row from np.eye(n) where x corresponds to the value of the flatten array (and do that for each value of the array)
    encoded_array = np.eye(n)[array.reshape(-1)]
    
    return encoded_array

# Load libraries
import math
import numpy as np
# Used to work with h5 files (designed to store and organize large amounts of data)
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.python.framework import ops
# import os


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


# Create a function to return mini-batches
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    # ------------------------------------------------------------------------- A CHECKER
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches (integer)
    seed -- make reproducible results

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    # Deterine the number of training examples
    m = X.shape[0] 
    # Create an empty list to return mini-batches
    mini_batches = []
    # Set the seed, to have reproducible results
    np.random.seed(seed)

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Partition shuffled_X and shuffled_Y to create complete mini_batches list containing tuples (mini_batch_X, mini_batch_Y)
    nb_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, nb_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k+1) * mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k+1) * mini_batch_size, :]
        # Create a tuple (mini_batch_X, mini_batch_Y)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handle the case of an incomplete mini-batch (when m % mini_batch_size != 0)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[nb_complete_minibatches * mini_batch_size : m, :, :, :]
        mini_batch_Y = shuffled_Y[nb_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# # Create the forward propagation of a 3 layers NN (for multiclass classification problem)
# def forward_propagation_for_predict(X, parameters):
#     """
#     Implement the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

#     Arguments:
#     X -- input dataset, of shape (input size, number of examples)
#     parameters -- python dictionary containing the parameters "W1", "b1", "W2", "b2", "W3", "b3"
#                   the shapes are given in initialize_parameters

#     Returns:
#     A3 -- the output of the last LINEAR unit
#     """
#     # Retrieve the parameters from the dictionary parameters
#     W1 = parameters['W1']
#     b1 = parameters['b1']
#     W2 = parameters['W2']
#     b2 = parameters['b2']
#     W3 = parameters['W3']
#     b3 = parameters['b3']
    
#     # Compute the linear transformation of the input data to produce the output of the 1st layer of a NN
#     Z1 = tf.add(tf.matmul(W1, X), b1)
#     # Apply the ReLU activation function
#     A1 = tf.nn.relu(Z1)

#     # Compute the linear transformation of the input data to produce the output of the 2nd layer of a NN
#     Z2 = tf.add(tf.matmul(W2, A1), b2)
#     # Apply the ReLU activation function
#     A2 = tf.nn.relu(Z2)

#     # Compute the linear transformation of the input data to produce the output of the 3rd layer of a NN
#     Z3 = tf.add(tf.matmul(W3, A2), b3)
#     # Apply the Softmax activation function for multi-class classification (last layer) (converts logits to probabilities)
#     A3 = tf.nn.softmax(Z3)

#     return A3


# # Create the function to predict
# def predict(X, parameters):
#     '''
#     xxx
    
#     Inputs:
#     X -- 
#     parameters -- 

#     Returns:
#     prediction -- 
#     '''
#     # Convert the parameters in tensor
#     W1 = tf.convert_to_tensor(parameters["W1"])
#     b1 = tf.convert_to_tensor(parameters["b1"])
#     W2 = tf.convert_to_tensor(parameters["W2"])
#     b2 = tf.convert_to_tensor(parameters["b2"])
#     W3 = tf.convert_to_tensor(parameters["W3"])
#     b3 = tf.convert_to_tensor(parameters["b3"])

#     # Create the params dictionnary
#     params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

#     # 
#     x = tf.placeholder("float", [12288, 1])

#     # Implement the forward propagation 
#     z3 = forward_propagation_for_predict(x, params)

#     # Get the proba
#     p = tf.argmax(z3)

#     sess = tf.Session()
#     prediction = sess.run(p, feed_dict={x: X})

#     return prediction

#########################################################################################################
#####                                                                                               #####
#####                      CONVOLUTIONAL NEURAL NETWORK - BINARY CLASSIFICATION                     #####
#####                                    Created on: 2025-02-20                                     #####
#####                                  Last updated on: 2025-03-13                                  #####
#####                                                                                               #####
#########################################################################################################

#########################################################################################################
#####                                  PACKAGES & GENERAL FUNCTIONS                                 #####
#########################################################################################################

# Clear the whole environment 
globals().clear()

# Load the libraries
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl

# Set up a seed for reproducible results
tf.keras.utils.set_random_seed(1)  
tf.config.experimental.enable_op_determinism()

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/image_classification')

# Load the general functions from the other script
from general_functions import *


#########################################################################################################
#####                              LOAD THE HAPPY DATASET & FIRST LOOK                              #####
#########################################################################################################

# Load X/Y train/test datasets and classes
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_h5_dataset(
    'datasets/train_happy.h5','datasets/test_happy.h5')

# Print the classes (0 for not happy and 1 for happy)
print(classes)

# Show a random image of each class to have a first look 
index_not_happy = np.random.choice(np.where(Y_train_orig[0] == 0)[0])
index_happy = np.random.choice(np.where(Y_train_orig[0] == 1)[0])

# Create a figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# Display the picture of an happy example
axes[0].imshow(X_train_orig[index_happy])
axes[0].set_title('Happy example (Y = '+str(Y_train_orig[0][index_happy])+')')
axes[0].axis('off')
# Display the picture of an not happy example
axes[1].imshow(X_train_orig[index_not_happy])
axes[1].set_title('Not happy example (Y = '+str(Y_train_orig[0][index_not_happy])+')')
axes[1].axis('off')
# Adjust layout for better spacing
plt.tight_layout()
plt.show()


#########################################################################################################
#####                                       DATA PREPARATION                                        #####
#########################################################################################################

# Normalize the image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape the Y vectors
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# Get the shapes of train/test sets
print ("Number of training examples = " + str(X_train.shape[0]))
print ("Number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


#########################################################################################################
#####                                  CREATE THE SEQUENTIAL MODEL                                  #####
#########################################################################################################

# Create the model with the following steps
# ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
model = tf.keras.Sequential([
    # ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
    tfl.ZeroPadding2D(padding=3, input_shape=(64,64,3)),
    # Conv2D with 32 7x7 filters and stride of 1
    tfl.Conv2D(filters=32, kernel_size=(7,7), strides=1),
    # BatchNormalization for axis 3 (ie the number of channels, also called the depth of the feature map)
    tfl.BatchNormalization(axis=3),
    # ReLU
    tfl.ReLU(),
    # Max Pooling 2D with default parameters
    tfl.MaxPool2D(),
    # Flatten layer
    tfl.Flatten(),
    # Dense layer with 1 unit for output & 'sigmoid' activation (since binary classification)
    tfl.Dense(units=1, activation='sigmoid')
])

# Compile the model with Adam optimizer, accuracy metric and binary crossentropy loss (classification pb)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Check the model's parameters (the layers type, the outputs shape, and the number of parameters for each layer)
model.summary()


#########################################################################################################
#####                                 TRAIN AND EVALUATE THE MODEL                                  #####
#########################################################################################################

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=16)
# accuracy: 0.98 - loss: 0.05 

# Evaluate the model on the test set (accuracy metric + binary crossentropy loss, as specified in the compile step)
model.evaluate(X_test, Y_test)
# accuracy:  0.89 - loss: 0.21 

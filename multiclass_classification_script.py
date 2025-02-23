#########################################################################################################
#####                                                                                               #####
#####                    CONVOLUTIONAL NEURAL NETWORK - MULTICLASS CLASSIFICATION                   #####
#####                                    Created on: 2025-02-20                                     #####
#####                                  Last updated on: 2025-02-23                                  #####
#####                                                                                               #####
#########################################################################################################

#########################################################################################################
#####                                  PACKAGES + GENERAL FUNCTIONS                                 #####
#########################################################################################################

# Clear the whole environment 
globals().clear()

# Load the libraries
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops

# Set up a seed for reproducible results
tf.keras.utils.set_random_seed(21)  
tf.config.experimental.enable_op_determinism()

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/image_classification')

# Load the general functions from the other script
from general_functions import *


#########################################################################################################
#####                              LOAD THE SIGNS DATASET + FIRST CHECK                             #####
#########################################################################################################

# Load X/Y train/test datasets
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_h5py_dataset(
    'datasets/train_signs.h5', 'datasets/test_signs.h5')

# Print the classes (from 0 to 5)
print(classes)

# Show an image of each class to have a first look (0-5)
index_0 = np.random.choice(np.where(Y_train_orig[0] == 0)[0])
index_1 = np.random.choice(np.where(Y_train_orig[0] == 1)[0])
index_2 = np.random.choice(np.where(Y_train_orig[0] == 2)[0])
index_3 = np.random.choice(np.where(Y_train_orig[0] == 3)[0])
index_4 = np.random.choice(np.where(Y_train_orig[0] == 4)[0])
index_5 = np.random.choice(np.where(Y_train_orig[0] == 5)[0])

# Create a figure with 6 subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(10, 5))
# Display the picture of an hand sign showing a 0
axes[0,0].imshow(X_train_orig[index_0])
axes[0,0].set_title('Hand sign '+str(Y_train_orig[0][index_0]))
axes[0,0].axis('off')
# Display the picture of an hand sign showing a 1
axes[0,1].imshow(X_train_orig[index_1])
axes[0,1].set_title('Hand sign '+str(Y_train_orig[0][index_1]))
axes[0,1].axis('off')
# Display the picture of an hand sign showing a 2
axes[0,2].imshow(X_train_orig[index_2])
axes[0,2].set_title('Hand sign '+str(Y_train_orig[0][index_2]))
axes[0,2].axis('off')
# Display the picture of an hand sign showing a 3
axes[1,0].imshow(X_train_orig[index_3])
axes[1,0].set_title('Hand sign '+str(Y_train_orig[0][index_3]))
axes[1,0].axis('off')
# Display the picture of an hand sign showing a 4
axes[1,1].imshow(X_train_orig[index_4])
axes[1,1].set_title('Hand sign '+str(Y_train_orig[0][index_4]))
axes[1,1].axis('off')
# Display the picture of an hand sign showing a 5
axes[1,2].imshow(X_train_orig[index_5])
axes[1,2].set_title('Hand sign '+str(Y_train_orig[0][index_5]))
axes[1,2].axis('off')
# Adjust layout for better spacing
plt.tight_layout()
plt.show()


#########################################################################################################
#####                                        DATA PREPARATION                                       #####
#########################################################################################################

# Normalize the image data
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# One-hot encode Y train and test and reshape
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# Get an idea of the shape of X/Y train/test datasets
print ("Number of training examples = " + str(X_train.shape[0]))
print ("Number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


#########################################################################################################
#####                                       CREATE THE MODEL                                        #####
#########################################################################################################

# Implement the cnn_model function to build the model with the following layers:
# CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE. 
def cnn_model(input_shape):
    """
    Implements the CNN model: 
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    
    # CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(filters=8,kernel_size=(4,4),strides=1,padding='same')(input_img)
    # RELU
    A1 = tfl.ReLU()(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tfl.MaxPool2D(pool_size=(8,8),strides=8,padding='same')(A1)
    # CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tfl.Conv2D(filters=16,kernel_size=(2,2),strides=1,padding='same')(P1)
    # RELU
    A2 = tfl.ReLU()(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tfl.MaxPool2D(pool_size=(4,4),strides=4,padding='same')(A2)
    # FLATTEN
    F =  tfl.Flatten()(P2)
    # DENSE: 6 neurons in output layer and softmax activation function (because multiclass classification)
    outputs = tfl.Dense(units=6,activation='softmax')(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

# Create the model using the inputs we have
model = cnn_model((64, 64, 3))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check the model's parameters (the layers type, the outputs shape, and the number of parameters for each layer)
model.summary()


#########################################################################################################
#####                                        TRAIN THE MODEL                                        #####
#########################################################################################################

# Transform the train dataset in mini-batches
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)

# Transform the test dataset in mini-batches
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)

# Train the model
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)


#########################################################################################################
#####                                       EVALUATE THE MODEL                                      #####
#########################################################################################################

# The history provides a record of all the loss and metric values in memory
history.history

# Determine the accuracy and loss of train/test sets over time
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)

# Visualize the accuracy and loss of train/test sets over time
# Create a figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# Display the accuracy with respect to epoches
axes[0].plot(df_acc)
axes[0].set_title('Model accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(['Train','Validation'],loc="upper left")
# Display the loss with respect to epoches
axes[1].plot(df_loss)
axes[1].set_title('Model loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(['Train','Validation'],loc="upper right")
# Adjust layout for better spacing
plt.tight_layout()
plt.show()

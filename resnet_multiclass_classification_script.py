##############################################################################################################
#####                                                                                                    #####
#####                          MULTICLASS CLASSIFICATION USING RESIDUALS NETWORK                         #####
#####                                       Created on: 2025-02-21                                       #####
#####                                       Updated on: 2025-02-26                                       #####
#####                                                                                                    #####
##############################################################################################################

##############################################################################################################
#####                                              PACKAGES                                              #####
##############################################################################################################

# Clear the whole environment
globals().clear()

# Load the libraries
import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# We set up a seed to have reproducible results
np.random.seed(1)
tf.random.set_seed(2)

# We set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/image_classification')
from general_functions import *


##############################################################################################################
#####                                         THE IDENTITY BLOCK                                         #####
##############################################################################################################

# The identity block is one of the 2 blocks used in ResNets. 
# It corresponds to the case where the input activation (a[l]) has the same dimension as the output activation (a[l+2]).

# We create the function identity_block to implement the ResNet identity block
def identity_block(X, f, filters, initializer=random_uniform):
    """
    Implementation of the identity block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    # If training is set to False, its weights are not updated with the new examples. I.e when the model is used in prediction mode.
    # axis: Integer, the axis that should be normalized (typically the features axis), here we are normalizing the channels axis
    X = BatchNormalization(axis = 3)(X) 
    X = Activation('relu')(X)
    
    ## Second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X) 
    X = Activation('relu')(X)

    ## Third component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X) 
    
    ## Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)

    return X


##############################################################################################################
#####                                      THE CONVOLUTIONAL BLOCK                                       #####
##############################################################################################################

# The ResNet convolutional block is the second typical block type. 
# The difference with the identity block is that there is a CONV2D layer in the shortcut path because it's used when input and output don't have the same dimensions

# Create the function implementing the convolutional block
def convolutional_block(X, f, filters, s = 2, initializer=glorot_uniform):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    # axis: Integer, the axis that should be normalized (typically the features axis), here we are normalizing the channels axis
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X) # no hyperparameters
    
    ## Second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding='same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)   

    ## Third component of main path (no ReLU activation function in this component)
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    
    ##### SHORTCUT PATH #####
    X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s,s), padding='valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    # Final step: the shortcut and the main path values are added together, then pass it through the ReLU activation function
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


##############################################################################################################
#####                             BUILDING THE RESNET MODEL (WITH 50 LAYERS)                             #####
##############################################################################################################

# Create the function implementing the ResNet model with 50 layers
def ResNet50(input_shape = (64, 64, 3), classes = 6, training=False):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    ## Stage 3
    X = convolutional_block(X, f = 3, filters = [128,128,512], s = 2)
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2))(X)

    # output layer
    X = Flatten()(X) # no hyperparameters
    # FC (= Dense) layer reduces its input to the number of classes
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model


##############################################################################################################
#####                                    TRAIN AND EVALUATE THE MODEL                                    #####
##############################################################################################################

# Create the model
model = ResNet50(input_shape = (64, 64, 3), classes = 6)

# Get the summary 
model.summary()

# Set up the optimizer with learning rate and compile the model
np.random.seed(1)
tf.random.set_seed(2)
opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Load the data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_h5py_dataset(
    train_path='datasets/train_signs.h5',test_path='datasets/test_signs.h5')

# Normalize the image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test Y to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# get an idea of the dimensions of train/test datasets
print ("Number of training examples = " + str(X_train.shape[0]))
print ("Number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# Train the model (with 20 epochs and only CPU, it takes around 1h)
model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

# Evaluate the model on test set
preds = model.evaluate(X_test, Y_test)
print ("Test Accuracy = " + str(preds[1])+" & Loss = " + str(preds[0]))


##############################################################################################################
#####                          USE A PRE-TRAINED MODEL AND TEST ON A NEW IMAGE                           #####
##############################################################################################################

# This ResNet50 model's weights were trained on the SIGNS dataset
pre_trained_model = load_model('resnet50.h5')

# Get a summary of the pre-trained model
pre_trained_model.summary()

# Evaluate the model on test set
preds = pre_trained_model.evaluate(X_test, Y_test)
print ("Test Accuracy = " + str(preds[1]) + " & Loss = " + str(preds[0]))

# Load an image, resize it, transform it to array
img = image.load_img('images/my_image.jpg', target_size=(64, 64))
x = image.img_to_array(img)

# Add an extra dimension (at the specified axis), done to convert a data array into a batch-like format (to use as input for ML model)
x = np.expand_dims(x, axis=0)

# Normalize it (pixel values will be between 0 and 1)
x = x/255.0

# Show the image and its dimensions
print('Input image shape:', x.shape)
plt.imshow(img)
plt.show()

# Get the prediction of the image (using the pre-trained model)
prediction = pre_trained_model.predict(x)
print(np.argmax(prediction)) # we get the index with highest probability


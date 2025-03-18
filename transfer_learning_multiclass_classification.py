##########################################################################################################
#####                                                                                                #####
#####                       MULTICLASS CLASSIFICATION USING TRANSFER LEARNING                        #####
#####                                     Created on: 2025-02-21                                     #####
#####                                     Updated on: 2025-03-18                                     #####
#####                                                                                                #####
##########################################################################################################

##########################################################################################################
#####                                             PACKAGES                                           #####
##########################################################################################################

# Clear the whole environment 
globals().clear()

# Load the libraries
import matplotlib.pyplot as plt
import json
import numpy as np
import setuptools.dist
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.layers  import RandomFlip, RandomRotation
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/image_classification')


##########################################################################################################
#####                    CREATE THE TRAINING AND VALIDATION DATASETS & FIRST LOOK                    #####
##########################################################################################################

# Training and validation sets parameters
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "datasets/"

# Read from the directory and create training dataset
train_dataset = image_dataset_from_directory(
    # Directory where the data is located
    directory,
    # Whether to shuffle the data
    shuffle = True,
    # Size of the batches of data
    batch_size = BATCH_SIZE,
    # Size to resize images to after they are read from disk, specified as (height, width)
    image_size = IMG_SIZE,
    # Fraction of data to reserve for validation (between 0 and 1)
    validation_split = 0.2,
    # Needs to be specified when validation_split is specified ("training" vs "validation")
    subset = 'training',
    # Must match each other, so the training and validation sets don't overlap
    seed = 42)

# Read from the directory and create validation dataset
validation_dataset = image_dataset_from_directory(directory,
                                                  shuffle = True,
                                                  batch_size = BATCH_SIZE,
                                                  image_size = IMG_SIZE,
                                                  validation_split = 0.2,
                                                  subset = 'validation',
                                                  seed = 42)

# Read the class names (alpaca and not alpaca)
class_names = train_dataset.class_names
print(class_names)

# Get a first look at 9 images from the train dataset
plt.figure(figsize=(10, 10))
# Iterate on the 1st batch of the train_dataset
for images, labels in train_dataset.take(1):
    # Take 9 images and display them in a 3x3 matrix
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # Convert to integer, as expected by plt.imshow()
        plt.imshow(images[i].numpy().astype("uint8"))
        # Display the class of the image as title on top of the image
        plt.title(class_names[labels[i]])
        plt.axis("off")
# Adjust layout for better spacing
plt.tight_layout()
plt.show()


##########################################################################################################
#####                              PREPROCESS AND AUGMENT TRAINING DATA                              #####
##########################################################################################################

# To automatically tune the number of parallel calls or workers used for data processing to improve performance without manually specifying values
AUTOTUNE = tf.data.experimental.AUTOTUNE

# To prevent memory bottlenecks that can occur when reading from disk
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

# Create the function data_augmenter for data augmentation using a Sequential Keras model composed of 2 layers:
# the RandomFlip('horizontal') layer and the RandomRotation(0.2) layer 
def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers: 
    * random horizontal (left-right) flip
    * random rotate within +/-72 degrees
    
    Returns:
    data_augmentation -- tf.keras.Sequential model
    '''
    # Create a sequential model
    data_augmentation = tf.keras.Sequential() 

    # Add a preprocessing layer which randomly flips images during training (50% chance of being left-right flipped)
    data_augmentation.add(RandomFlip('horizontal'))

    # Add a preprocessing layer which randomly rotates images during training (by a random amount in the range [-20% * 2pi, 20% * 2pi], which corresponds to +/-72)
    data_augmentation.add(RandomRotation(0.2))
    
    return data_augmentation


# Store the sequential model (for data augmentation)
data_augmentation = data_augmenter()

# Get a first look at the data augmentation carried on on our training data (in this example with 11 new images)
# Iterate on the 1st batch (we ignore the label because it's not needed for this visualisation)
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    # Select randomly an image
    first_image = image[np.random.randint(len(image))]
    # Iterate 11 times, to get 11 augmented versions of the image
    for i in range(11):
        # Get a 3*4 grid
        ax = plt.subplot(3, 4, i + 1)
        # Apply the data_augmentation model to the image selected (after considering it as a bactch of 1 image)
        augmented_image = data_augmentation(tf.expand_dims(first_image, axis=0))
        # Display the augmented image (normalized because format expected by imshow)
        plt.imshow(augmented_image[0]/255)
        plt.axis('off')
    # Show the original image at the bottom right (12th picture)
    ax = plt.subplot(3, 4, 12)
    plt.imshow(first_image/255)
    plt.title('original image')
    plt.axis('off')
plt.show()

# Prepare input data for the MobileNetV2 model by performing specific preprocessing steps that are required for the model to work correctly
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


##########################################################################################################
#####                                       MOBILENETV2 MODEL                                        #####
##########################################################################################################

# Set up the image shape concatenating the image size and 3 (for 3 channels since we use RGB images)
IMG_SHAPE = IMG_SIZE + (3,)

# Load the weights we will use (from a pre-trained model)
base_model_path = "models/with_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5"

# Set the model (use pre-trained weights)
base_model = tf.keras.applications.MobileNetV2(
    # Specify the shape of our images
    input_shape = IMG_SHAPE,
    # Use all the layers from the pretrained model
    include_top = True,
    # Specify the weights we want to use
    weights = base_model_path)

# Print the model summary (to see all the model's layers, the shapes of their outputs, and the total number of parameters, trainable and non-trainable)
base_model.summary()

# Get the number of layers
nb_layers = len(base_model.layers)
print(nb_layers)

# The last 2 layers (called top layers) are responsible of the classification in the model
print(base_model.layers[nb_layers - 2].name)
print(base_model.layers[nb_layers - 1].name)


# Choose the first batch from the tensorflow dataset (32 images since batch size is 32)
image_batch, label_batch = next(iter(train_dataset))
# Run it through the MobileNetV2 base model to test out the predictions on some of our images 
feature_batch = base_model(image_batch)
# We get, for each image, the probability for each of the 1000 classes (the base model is based on)
print(feature_batch.shape)

# Show the correct label for the batch we ran the model on
label_batch

# Model's weights are frozen since we used a model that was already pre-trained
base_model.trainable = False
# Pre-process the image_batch (performing specific preprocessing steps that are required for the model to work correctly) 
# Transform to tf.Variable (to create a mutable tensor that can be updated)
image_var = tf.Variable(preprocess_input(image_batch))
# Pass the first image batch through the model
pred = base_model(image_var)

# Create decode_predictions function to decode predictions: we get the best 2 predictions, leading to the best 2 classes 
def decode_predictions(preds, top=2):
    ''''
    Get the best 2 predictions, leading to the best 2 classes

    Inputs:
    preds -- array of predictions
    top -- number of classes we want (integer)

    Returns:
    results -- tuple with the class of the 2 indexes with highest probability (highest, 2nd highest)
    '''
    # Create an empty list
    results = []

    # Iterate on preds (each image of the batch)
    for pred in preds:
        # Get the index of the 2 highest predictions, with the 1st index having the highest probability
        top_indices = pred.argsort()[-top:][::-1]
        # Get the class of each index for each index (with the highest probability)
        result = [tuple(class_index[str(i)]) + (pred[i],) for i in top_indices]
        # Append the result of this image to the results list
        results.append(result)

    return results


# Get all the classes the model MobileNetV2 was trained on
with open("models/imagenet_class_index.json", 'r') as f:
    class_index = json.load(f)
print(class_index)


# Get the best 2 classes for each image of the batch
decoded_predictions = decode_predictions(pred.numpy(), top=2)
print(decoded_predictions)
# We noticed that no labels mention "alpaca" and "not alpaca" (while this was our goal)
# This is because we used the model MobileNetV2, which was trained on 1000 classes (with no "alpaca" and "not alpaca" classes)
# So we are going to delete the last/top 2 layers to create a new classification layer


##########################################################################################################
#####                      MOBILENETV2 MODEL CHANGING THE CLASSIFICATION LAYER                       #####
##########################################################################################################

# Create a model using the pre-trained MobileNetV2 model but changing the top layers for a binary classification (to recognize alpacas)
def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    '''
    Define a tf.keras model for binary classification out of the MobileNetV2 model
    
    Inputs:
    image_shape -- image width and height
    data_augmentation -- data augmentation function
    
    Returns:
    model -- tf.keras.model
    '''
    # Set up the shape of the inputs
    input_shape = image_shape + (3,)
    
    # Load the weights we will use (from a pre-trained model)
    base_model_path  = "models/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"

    # Set the model (use pre-trained weights)
    base_model = tf.keras.applications.MobileNetV2(
        # Specify the shape of our images
        input_shape = input_shape,
        # Use all the layers from the pretrained model except the top 2
        include_top = False,
        # Specify the weights we want to use
        weights = base_model_path)
    
    # Freeze the base model by making it non trainable (transfer learning)
    base_model.trainable = False 

    # Create the input layer (same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape = input_shape) 
    
    # Apply data augmentation to the inputs (to have "more" images on the same number of real images)
    X = data_augmentation(inputs)

    # Pre-process the image (performing specific preprocessing steps that are required for the model to work correctly) 
    X = preprocess_input(X) 
    
    # training parameter is set to False to avoid keeping track of statistics in the batch norm layer
    X = base_model(X, training=False) 
    
    # Add the new binary classification layers (use global avg pooling to summarize the information in each channel)
    X = tfl.GlobalAveragePooling2D()(X)  

    # Include dropout with probability of 0.2 (to avoid overfitting)
    X = tfl.Dropout(0.2)(X)

    # Use a prediction layer with one neuron (as a binary classifier only needs one)
    prediction_layer =  tfl.Dense(1, activation = 'linear')
    outputs = prediction_layer(X) 

    # Create the model
    model = tf.keras.Model(inputs, outputs)
    
    return model


# Create our new model using the data_augmentation function defined earlier
model2 = alpaca_model(IMG_SIZE, data_augmentation)

# Compile the model
base_learning_rate = 0.001
model2.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = base_learning_rate),
               loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics = ['accuracy'])

# Set up the number of epoches (number of times the entire training dataset is passed through the model during the training process)
initial_epochs = 5

# Train the model with our train and validation sets
history = model2.fit(train_dataset, validation_data = validation_dataset, epochs = initial_epochs)

# Get the training and validation accuracy/loss
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']
loss = [None] + history.history['loss']
val_loss = [None] + history.history['val_loss']

# Show the training and validation accuracy/loss (with 2 plots, one over the other)
plt.figure(figsize=(8, 8))
# Plot the accuracy
ax1=plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
# Plot the loss
ax2=plt.subplot(2, 1, 2, sharex = ax1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()

# Print the class_names (correct for our goal)
print(class_names)

# The model is predicting between alpacas and not alpacas but accuracy is not that good, so let's try some fine-tuning


##########################################################################################################
#####               MOBILENETV2 MODEL CHANGING THE CLASSIFICATION LAYER & FINE-TUNING                #####
##########################################################################################################

# Get a look at the summary and all layers from model2
model2.summary()
model2.layers

# Determine the layer to fine-tune from 
base_model = model2.layers[2]

# Unfreeze the base_model (to be able to update the model's weights during the training)
base_model.trainable = True

# Take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 120

# Freeze all the layers before the fine_tune_at layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model with BinaryCrossentropy loss function, Adam optimizer (with a learning rate of 0.1 * base_learning_rate) and accuracy metric
model2.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
               optimizer = tf.keras.optimizers.Adam(learning_rate=0.1*base_learning_rate),
               metrics = ['accuracy'])

# Decide on the numer fo epochs to use to fine-tune
fine_tune_epochs = 50

# Determine the total number of epochs (by adding the number of epochs used for fine tuning)
total_epochs =  initial_epochs + fine_tune_epochs

# Train the model with fine-tuning
history2 = model2.fit(
    # Train set
    train_dataset,
    # Total number of epoches to train on
    epochs = total_epochs,
    # Continue the training from where the previous training session left off
    initial_epoch = history.epoch[-1],
    # Validation set
    validation_data = validation_dataset)

# Determine the accuracy and loss on training and validation sets (we complete the list of history results with the fine-tune history we just got)
acc += history2.history['accuracy']
val_acc += history2.history['val_accuracy']
loss += history2.history['loss']
val_loss += history2.history['val_loss']

# Plot the training/validation accuracy/loss
plt.figure(figsize=(8, 8))
# Plot the accuracy
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
# Plot a green dash line to highlight when the fine tuning starts
plt.plot([initial_epochs+1,initial_epochs+1], plt.ylim(), ':',label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
# Plot the loss
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
# Plot a green dash line to highlight when the fine tuning starts
plt.plot([initial_epochs+1,initial_epochs+1], plt.ylim(), ':',label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# Convolutional Neural Network
In this repository, we are going to carry several image classification projects:
* a binary classification problem (using a TF Keras Sequential API) 
* a multiclass classification problem (using the TF Keras Functional API)
For this projects we mainly use the framework Tensorflow - Keras (it has pre-defined layers that allows for more simplified and optimized model creation and training)


## Binary classification (happy vs not happy)
### Project description
In this project, we aim at classifying an image into 2 classes: happy or not happy. To do so, we will build a CNN model that determines whether the people in the images are smiling or not. The business use could be: people can only enter a house/shop if they are smiling.

### Datasets
We use the Happy House dataset which contains images of peoples' faces. The training set contains 600 images and the test set contains 150 images. Images are 64x64 pixels in RGB format (so 3 channels).

### Model used
For this binary easy classification, we are going to use the Sequential API.
It allows us to build a model layer by layer. It's ideal for building models where each layer has exactly one input tensor and one output tensor. 
It's simple and straightforward (but only appropriate for simple models with layer operations that proceed in a sequential order [like a Python list])

### Script description, step by step
1. We load the libraries that we will need for the project, we load the general functions built in another script and we set up a seed to get reproducible results.
2. We load the training and test sets and we take a look at the classes (0-1 for not happy-happy) and 2 pictures (1 for each class)
3. We prepare a bit the data (normalization of the X data [to get faster convergence of the model] and transpose to have the right shape of Y data) and we check the dimensions of each set.
4. We create the sequential model and compile it
5. We train and evaluate the model


## Multiclass classification (sign language digits)
### Project description
In this project, we aim at classifying an image into 6 classes: 0, 1, 2, 3, 4 or 5. To do so we will build a CNN model that determines the number showed by the the hand in the images.
The business use could be: being able to recognize language signs (then we would need to extend that problem to all language signs).

### Datasets
We use the Signs dataset which contains images of hands showing a number (from 0 to 5). The training set contains 1080 images and the test set contains 120 images. Images are 64x64 pixels in RGB format (so 3 channels).

### Model used
The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs (it allows more flexibility). 
In this example we will use Functional API just to know how to use it, eventhough the model is actually linear so we could have used the Sequential API.
Steps:
1. We create the input node
2. We create the next layer using one of the previous object/input as input and we save it to another object
3. We continue until creating the output 
4. Finally we create the model taking the input and output

### Script description, step by step
1. We load the libraries that we will need for the project, we load the general functions built in another script and we set up a seed to get reproducible results.
2. We load the training and test sets and we take a look at the classes (0-5) and 6 pictures (1 for each class)
3. We prepare a bit the data (normalization of the X data [to get faster convergence of the model] and transpose to have the right shape of Y data) and we check the dimensions of each set.
4. We create the model and compile it
5. We train using mini-batches
6. We evaluate the model


## Sequential vs Functional APIs
Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model.
For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. 

### Different layers
These are the several layers we can use to build a Sequential or Functional API: 
* Conv2D: creates a convolution kernel that is convolved with the layer input over a 2D spatial dimension (height and width)
* MaxPool2D: downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size (f, f)) for each channel of the input. The window is shifted by strides (s, s) along each dimension.
* ReLU: computes the elementwise ReLU (Rectified Linear Unit)
* Flatten: given a tensor, this function takes each example in the batch and flattens it into a 1D vector. If the tensor has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), with k=$h\times c$
* Dense: given a flattened input, it returns the output computed using a fully connected layer. 

Pool size and kernel size refer to the same thing in different objects: they refer to the shape of the window where the operation takes place (MaxPool2D vs Conv2D). 

Before creating the model we need to define the output (for example Dense layer with as many units as output classes and activation softmax in the case of a multiclass classification or sigmoid in the case of binary classification)
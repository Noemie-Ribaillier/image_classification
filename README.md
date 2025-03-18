# Image classification
In this repository, we are going to do several image classification projects:
* a binary classification problem (using a TF Keras Sequential API) 
* a multiclass classification problem (using the TF Keras Functional API)
* a multiclass classification problem (using the ResNet50 model)
* a binary classification problem (using transfer learning)
For these projects we use the framework Tensorflow - Keras (it has pre-defined layers that allow for more simplified and optimized model creation and training).

 
## Binary classification (happy vs not happy, using TF Keras Sequential API)
### Project description
In this project, we aim at classifying an image into 2 classes: happy (value 1) or not happy (value 0). To do so, we will build a CNN model that determines whether the person in the image is smiling or not. The business use could be: people can only enter a house/shop if they are smiling.

### Dataset
We use the Happy House dataset which contains images of peoples' faces. The training set contains 600 images and the test set contains 150 images. Images are 64x64 pixels in RGB format (so 3 channels).

### Model used
For this binary easy classification, we are going to use the Sequential API.
It allows us to build a model layer by layer. It's ideal for building models where each layer has exactly one input tensor and one output tensor. 
It's simple and straightforward (but only appropriate for simple models with layer operations that proceed in a sequential order [like a Python list])

### Script description, step by step
1. We load the libraries that we will need for the project, we load the general functions built in another script and we set up a seed to get reproducible results.
2. We load the training and test sets and we take a look at the classes (0-1 for not happy-happy) and 2 pictures (1 for each class)
3. We prepare the data (normalization of the X data and transpose to have the right shape of Y data) and we check the dimensions of each set. The normalization of the X has several goals:
* Get faster and more stable convergence of the model
* Get consistancy among datasets (if pixel ranges are not the same among images, normalization solve this issue, which makes it easier for models to generalize across datasets)
* Prevent dominance of features with larger magnitudes (normalization ensures that no single feature dominates during training)
* Form of regularization (because it keeps input values in a controlled range), reducing the chances of overfitting in some cases, especially when combined with techniques like batch normalization.
4. We create the sequential model and compile it. For this model we use the following layers:
* ZeroPadding2D: adds padding around the input image to prevent the reduction of spatial dimensions after convolution (and pooling operations). This helps maintain the spatial dimensions and can also help prevent edge information from being lost during convolution.
* Conv2D: applies a convolution operation to the input data. Convolution layers are used to automatically learn spatial features from the input image, such as edges, textures, and patterns.
ZeroPadding2D and Conv2D could also be summed up as Conv2D with parameter padding='same'.
* BatchNormalization: normalizes the output of the previous layer to improve model training. It helps speed up training and may lead to better performance by ensuring that the activations maintain a consistent distribution throughout training.
* ReLU activation function: introduces non-linearity into the model and helps the network learn complex patterns.
* Max Pooling 2D: used to reduce the spatial dimensions (height and width) of the input, effectively downsampling the feature maps. This reduces the computational load and the number of parameters.
* Flatten: flattens the multi-dimensional input (eg, the 2D feature maps from the convolution layers) into a 1D vector. This is necessary because the next layer (the Dense layer) expects a 1D vector as input.
* Dense (with sigmoid activation function): used to make predictions. It uses a fully connected (dense) layer with a sigmoid activation function to output a probability for binary classification.
5. We train and evaluate the model


## Multiclass classification (sign language digits, using the TF Keras Functional API)
### Project description
In this project, we aim at classifying an image into 6 classes: 0, 1, 2, 3, 4 or 5 (according to the number displayed by the hand). To do so we will build a CNN model that determines the number showed by the hand in the image.
The business use could be: being able to recognize language signs (then we would need to extend that project to all language signs).

### Dataset
We use the Signs dataset which contains images of hands showing a number (from 0 to 5). The training set contains 1080 images and the test set contains 120 images. Images are 64x64 pixels in RGB format (so 3 channels).

### Model used
The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs (it allows more flexibility). 
In this example we will use Functional API just to know how to use it, eventhough the model is actually linear so we could have used the Sequential API.
Steps:
1. Create the input node
2. Create the next layer using one of the previous object/input as input and save it to another object
3. Continue until creating the output 
4. Finally create the model taking the input and output

### Script description, step by step
1. We load the libraries that we will need for the project, we load the general functions built in another script and we set up a seed to get reproducible results.
2. We load the training and test sets and we take a look at the classes (0-5) and 6 pictures (1 for each class)
3. We prepare the data (and we check the dimensions):
* normalization of the X data: to get faster and more stable convergence of the model, get consistancy among datasets, prevent dominance of features with larger magnitudes and used as a form of regularization 
* transform Y data to one-hot encoding: to use softmax as final activation function (to get the probability of each class)
4. We create the model and compile it. For this model, we use the following layers:
* Conv2D: to learn spatial features from the input image, such as edges, textures, and patterns. By using same padding, we ensure that the features at the edges of the image are preserved, so the model doesn't lose important information from the boundaries.
* ReLU activation function: to introduce non-linearity into the model and helps the network learn complex patterns.
* MaxPool: to retain the most important features, here we use same padding (to keep as much edges information as possible). The spatial dimensions (height and width) are reduced since the number of strides is different than 1 (typically spatial dimensions are divided by strides number).
* then we repeat it: the 1st round of Conv2D, ReLU, MaxPool extracts low level features (like edges, textures, and simple shapes), then the 2nd round extracts more complex features (like patterns, corners, and parts of objects). Additionally, the number of filters increases, allowing the network to learn richer and more abstract features as we go deeper.
* Flatten: flattens the multi-dimensional input (eg, the 2D feature maps from the convolution layers) into a 1D vector. This is necessary because the next layer (the Dense layer) expects a 1D vector as input.
* Dense (with softmax activation function and 6 units): used to make predictions. The softmax activation ensures that the output values sum to 1, representing the probabilities of each of the 6 classes in a multiclass classification task.
5. We train using mini-batches (splitting the initial set into smaller samples) for:
* Memory constraints: datasets, especially in deep learning, can be very large, sometimes even too large to fit into memory all at once. By splitting the data into smaller mini-batches, we can process chunks of data that fit into memory, making training more feasible.
* Faster computations: instead of processing the entire dataset at once, mini-batches allow the model to update parameters after each batch, which speeds up the learning process compared to waiting for the entire dataset.
* Updating gradients: when training models, we want to update the model's parameters based on the gradients of the loss function. Using the entire dataset (full-batch gradient descent) would compute the gradients over all data, which is computationally expensive. On the other hand, mini-batch gradient descent computes gradients for a smaller batch, offering a balance between the efficiency of full-batch and the randomness of stochastic gradient descent (SGD).
* Noise and regularization: by using mini-batches, the model sees different subsets of the data in each step, which introduces noise in the gradient updates. This noise helps the model avoid overfitting to the training data and can improve its ability to generalize to unseen data.
* Regularization effect: this inherent randomness in the process can have a regularization effect, making the model less likely to overfit to the training set, leading to better performance on the test set.
* GPU and hardware optimization: mini-batches allow for parallel computation. Modern hardware, especially GPUs, is optimized for matrix operations on batches of data. By feeding mini-batches into the hardware, computations are done much faster compared to sequential updates. This can lead to significant speedups in training.
* For the test set, while mini-batching is not strictly necessary (because we usually process the entire test set at once), it can still be used to avoid memory issues with very large datasets, particularly in real-world scenarios where the test data may be too large to load in memory all at once.
6. We evaluate the model


## Sequential vs Functional APIs
Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model.
For any layer construction in Keras, we'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. 

### Different layers
These are the several layers we can use to build a Sequential or Functional API: 
* Conv2D: creates a convolution kernel that is convolved with the layer input over a 2D spatial dimension (height and width). It performs a sliding window operation to capture local spatial features (such as edges, textures, and patterns) from the input image. The result is a set of feature maps.
* MaxPool2D: take the maximum value over an input window (of size defined by pool_size (f, f)) for each channel of the input. The window is shifted by strides (s, s) along each dimension. It reduces the spatial size of the input, helping to control overfitting and reduce computational complexity. Same padding with strides=1 ensures that spatial dimensions are the same between input and output. 
* ReLU: computes the elementwise ReLU (Rectified Linear Unit), it introduces non-linearity to the network, allowing it to learn more complex patterns.
* Flatten: given a tensor, this function takes each example in the batch and flattens it into a 1D vector. If the tensor has the shape (batch_size, h, w, c), it returns a flattened tensor with shape (batch_size, k), with k = h * w * c
* Dense: given a flattened input, it returns the output computed using a fully connected layer. 

Pool size and kernel size refer to the same thing in different objects: they refer to the shape of the window where the operation takes place (MaxPool2D vs Conv2D). 

Before creating the model we need to define the output (for example Dense layer with as many units as output classes and activation softmax in the case of a multiclass classification or sigmoid with one unit in the case of binary classification).


## Multiclass classification (sign language digits using ResNet50 model)
### Project description
In this project, we aim at classifying an image into 6 classes: 0, 1, 2, 3, 4 or 5 (according to the number displayed by the hand). To do so, we will build a ResNet50 model that determines the number showed by the hand in the image.
The business use could be: being able to recognize language signs (then we would need to extend that project to all language signs).

### Dataset
We use the Signs dataset which contains images of hands showing a number (from 0 to 5). The training set contains 1080 images and the test set contains 120 images. Images are 64x64 pixels in RGB format (so 3 channels).

### Model used
We use the ResNet50 model for this project. 

The main benefit of a very deep network is that it can learn features at many different levels of abstraction, from edges (at the shallower layers, closer to the input) to very complex features (at the deeper layers, closer to the output). However, deep network face the vanishing gradients problem: gradient signal that goes to zero quickly, thus making gradient descent prohibitively slow. During gradient descent, as we backpropagate from the final layer back to the first layer, we are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" from gaining very large values). We are going to build a deep convolutional network using Residual Networks (ResNets) with 50 layers. 

ResNet model learns residuals, meaning it tries to model the difference between the input and the desired output, instead of learning the entire output from scratch. Indeed R(x)=output-input=H(x)-x so H(x)=R(x)+x (meaning the residual block is overall trying to learn the true output H(x)). To summarize, the layers in a traditional network are learning the true output (H(x)), whereas the layers in a residual network are learning the residual (R(x)). Hence, the name: Residual Block.
ResNet model uses residual blocks, of 2 types: identity block and convolutional block.

The approach is to add a shortcut (also called skip connection) that allows information to flow more easily from one layer to the next's next layer. 2 advantages of residual block:
* Adding additional layers would not hurt the model's performance as regularisation will skip over them if those layers were not useful
* If the additional layers were useful, even with the presence of regularisation, the weights or kernels of the layers will be non-0 and model performance could increase slightly
The skip connection (used in both blocks) helps address the vanishing gradient problem by enabling gradients to flow more directly through the network, allowing for more effective training of very deep networks. Indeed, with skip connections, gradients can directly flow through the shortcut path, bypassing some layers. This makes the gradients less likely to shrink as they pass through many layers, thus mitigating the vanishing gradient problem.

We will also use the bottleneck residual block structure to improve efficiency and reduce computational complexity since we use a very deep ResNet.

#### Identity block
The goal of the identity block is to keep the input unchanged (identity) and adds it to the output of the convolutional layers (residual learning). 
Since the input and output dimensions are the same, the identity block typically learns small transformations (if any). It can learn to refine features, but it doesn't drastically alter the input's structure or size.

The identity block is used when the input activation has the same dimensions than the output activation. So there is no change in spatial dimensions (height, width) or number of channels between input and output. The input is directly added to the output of the convolutional layers (skip connection). 

The identity block uses a shortcut path. Each path gets the following layers:
* the main path: Conv2D -> BN -> ReLu -> Conv2D -> BN -> ReLu -> Conv2D -> BN
* the shortcut path is "empty" since dimensions are exactly the same between input and output
BatchNorm step is added to stabilize and accelerate training by normalizing the input to the following layer.

#### Convolutional block
The convolutional block applies transformations that can change the spatial dimensions (height, width) or number of channels. It typically learns larger transformations (which can be critical for capturing more abstract representations of the data) by:
* Downsampling (ie spatial size is reduced): using convolutions with stride different than 1 to reduce the spatial resolution of the feature maps while trying to retain relevant features
* Increasing the depth (ie number of channels): using convolutions with more filters to capture more complex features

The convolutional block uses a shortcut path. Each path gets the following layers:
* the main path: Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU -> Conv2D -> BN
* the shortcut path: Conv2D -> BN. The Conv2D layer in the shortcut path is to apply a (learned) linear function used to resize the input X to a different dimension so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. 
BatchNorm step is added to stabilize and accelerate training by normalizing the input to the following layer.

#### Bottleneck blocks
Both identity and convolutional blocks here follow the typical structure of bottleneck residual blocks.

We use bottleneck here to improve efficiency and reduce computational complexity (especially for ResNet with 50 and more layers). The term "bottleneck" comes from the fact that it reduces the number of channels in the input tensor before applying the main convolution operation and then restores the number of channels afterwards (or more). 
The bottleneck residual block has the following strucutre:
* A 1x1 convolutional layer that reduces the number of channels (filters) in the input tensor. This layer is responsible for creating the bottleneck
* A convolutional layer that performs the main convolution operation on the reduced number of channels. This layer is responsible for learning spatial features and features extraction 
* A 1x1 convolutional layer that restores the number of channels to the original number or increase it. This layer is responsible for expanding the output back to the original dimension (or more)

Advantage of the bottleneck:
* Reduced computational complexity: the bottleneck design reduces the number of operations required, making the model more efficient and faster to train
* Better parameter efficiency: by reducing the number of channels in intermediate layers, bottleneck residual blocks utilize fewer parameters, leading to a more compact model
* Preservation of representational capacity: despite the reduction in computational complexity and parameters, the bottleneck design maintains the representational capacity of the original residual block

#### Details of the ResNet50
These are the steps used to build the ResNet50 model:
* Zero-padding with (3,3)
* Stage 1:
    * Conv2D with 64 filters, (7,7) kernel and (2,2) stride 
    * BatchNorm (applied to the channels axis of the input)
    * ReLU
    * MaxPool, (3,3) window and (2,2) stride
* Stage 2:
    * Convolutional block with 3 filters of size [64,64,256], (3,3) kernel and (1,1) stride 
    * 2 identity blocks with 3 filters of size [64,64,256], (3,3) kernel of middle Conv2D for the main path
* Stage 3:
    * Convolutional block with 3 filters of size [128,128,512], (3,3) kernel and (2,2) stride 
    * 3 identity blocks with 3 filters of size [128,128,512], (3,3) kernel of middle Conv2D for the main path
* Stage 4:
    * Convolutional block with 3 filters of size [256,256,1024], (3,3) kernel and (2,2) stride 
    * 5 identity blocks with 3 filters of size [256,256,1024], (3,3) kernel of middle Conv2D for the main path
* Stage 5:
    * Convolutional block with 3 filters of size [512,512,2048], (3,3) kernel and (2,2) stride 
    * 2 identity blocks with 3 filters of size [512,512,2048], (3,3) kernel of middle Conv2D for the main path
* AveragePool2D with (2,2) pool size
* Flatten layer (no hyperparameters)
* Fully Connected (Dense) layer: reduces the input to the number of classes using a softmax activation.

### Script description, step by step
1. We load the libraries that we will need for the poject, we load the general functions built in another script and we set up a seed to get reproducible results
2. We build the identity block function
3. We build the convolution block function
4. We build the ResNet50 model using the identity block and the convolution block
5. We train the model and evaluate on test set
6. We us a pre-trained model and we test it on a new image


## Comparison of the models (used for multiclass classification on signs dataset)
We find better results with ResNet50 model (especially the pre-trained one) than with the standard CNN for multiclass classification. It's because the ResNet50 model is deeper (but handle the vanishing gradient) and traiend on more images/epoches.


## Binary classification (using transfer learning)
### Project description
In this project, we aim at classifying an image into 2 classes: 'alpaca' vs 'not alpaca'. To do so, we will use transfer learning (ie use a model that was pre-trained) and for this specific project we will use the MobileNetV2 model.

### Dataset
We use a dataset gathering images of alpacas and images of non alpacas (already split into 2 folders). We have 142 pictures of alpacas and 185 pictures of non alpacas (so a total of 327 images). We create the train and validation set by putting 20% of all the images into the validation set. So the train set contains 262 images and the validation set contains 65 images. Not all images have the same shape, that's why they are, later in the script, resized (to all have the same shape) to have a height and a width of 160. Images are in RGB format (so data has 3 channels). 


#### Dataset preprocessing
Training set is pre-processed for this specific model to work correctly. We carry on data augmentation on the train set to increase the training set without increasing the number of images:
* random left-right flip 
* random rotation within +/-72 degrees


### Transfer Learning
Transfer learning uses a model, already pretrained on a large dataset (MobileNetV2 in this case, pre-trained on ImageNet dataset) and saved, to customize our own model cheaply and efficiently.
To adapt the classifier to new data, we delete the top/last layers, add a new classification layers, and train only on that layer. We replace the top layer to adapt the model to the specific classification task (because the pre-trained model already has learned useful feature representations but needs a new head for the specific target classes).
For this project we do the 3 following modelling:
* Use the pre-trained MobileNetV2 model as it is (so trained on 1000 classes, not the classes of interest for this project)
* Use the pre-trained MobileNetV2 model, removing the last/top 2 layers to make the pre-trained MobileNetV2 model efficient on the classes of this project 
* Fine-tune several late layers to make the model better at classifying our project


### More details about the MobileNetV2 model
MobileNetV2 model has been trained on ImageNet dataset (trained on 1.2 million images and 1000 classes). It provides fast and computationally efficient performance for object detection, image segmentation and image classification. The goal of MobileNetV2 is to balance between maintaning performance and reducing computational cost. It is optimized to run on mobile and other low-power applications. 

The architecture has 4 defining characteristics (gathered in inverted residual blocks):
* Depthwise separable convolutions: used to reduce the computational load while maintaining accuracy. They separate the convolution process into two steps: depthwise convolutions (which apply filters to each input channel separately) and pointwise convolutions (which combine the depthwise outputs)
* Linear bottlenecks: they help preserve the information by maintaining low-dimensional representations, which minimize information loss and improve the overall accuracy of the model
* Shortcut connections between bottleneck layers: when the initial and final feature maps are of the same dimensions (when the depthwise convolution stride equals one and input and output channels are equal), a residual connection is added to aid gradient flow during backpropagation
* ReLU6 activation function: it clips the ReLU output at 6. This helps prevent numerical instability in low-precision computations making the model more suitable for mobile and embeddded devices used because of its robustness when used with low-precision computation. It is used to prevent extremely large values during the forward pass, especially in situations like mobile devices where limiting large values can help improve performance and reduce resource consumption (like memory and computation). It also helps prevent the exploding gradients problem during training, where gradients become too large, causing instability in learning.


#### The depthwise separable convolutions
They are able to reduce the number of trainable parameters and operations and also speed up (traditional) convolutions in 2 steps: 
1. The depthwise convolution: it applies a separate filter for each input channel, reducing the computational cost. The filter/kernel is applied on each filter independently
2. The pointwise convolution: a 1x1 convolution combining the outputs of the previous step into one (combining these feature across channels to capture relationships between them). This gives a single result from a single feature at a time, and then is applied to all the filters in the output layer

Reminder over "normal" convolutions: 
* The kernel has the same depth has the input matrix (except for special cases such as 3D convolutions in medical field)
* The output will have the depth equal to the number of filters we choose

Difference between "normal" convolution and depthwise-separable convolution on a specific example (input matrix has shape 12x12x3 and kernel has shape 5x5x3 with 256 filters):
* "normal" convolution: we slide a filter (kernel) over the input image, performing element-wise multiplication, and summing the results to produce a single output value. For an input shape of 12x12x3 and a kernel shape of 5x5x3, the output size would be reduced to 8x8 because the kernel processes 5x5 regions of the 12x12 input and moves across the image with a stride (let's consider 1 here), applying the filter at each position.
This makes a total of 5 * 5 * 3 computations (for 1 kernel over 1 "sub-region") * 8 * 8 (for 1 kernel over all sub-regions) * 256 (for all filters), so a total of 1228800 computations
* depthwise-separable convolution (split in 2 steps):
    * depthwise convolution: each channel of the kernel is applied on the right channel of the input, giving the output with a split on the input filter. So in our example, it makes 5 * 5 computations (for 1 input channel) * 3 (for 3 input channels) * 8 * 8 (for the whole output matrix, sliding the whole input matrix)
    * pointwise convolution: we use a 1x1 (3rd dimension is the number of input channel, so here 3) kernel. We slide it over the temporary matrix, performing element-wise multiplication, and summing the results to produce a single output value. For the temporary shape of 8x8x3 and a kernel shape of 1x1x3, the output size becomes 8x8x256 because the kernel moves across the temporary image with a stride of 1 and we use 256 filters, applying the filter at each position. This adds 3 (1x1 convolution for each input channel) * 8 * 8 (each position of the output matrix) * 256 (all filters) computations
    * combining both steps, we end up with a total of 4800+49152=53952 computations (with depthwise separable convolution for this example)
So the number of parameters (to train) is very different between the depthwise separable convolution and the "normal" convolution. This separable approach significantly reduces the number of trainable parameters and computations which lowers the computational cost with only a small reduction in accuracy (that's why MobileNetV2 is so efficient).


#### The linear bottleneck layer
Each block consists of an inverted residual structure with a bottleneck at each end. These bottlenecks encode the intermediate inputs and outputs in a low dimensional space, and prevent non-linearities from destroying important information. 
Indeed, applying ReLU (non-linear transformation) in low-dimensional spaces can result in lost features/information loss. MobileNetV2 use linear bottlenecks to ensure that important features are preserved during the compression and expansion processes, helping the model to remain accurate without adding too much computational cost.


#### The shortcut connection
They are similar to the ones in traditional residual networks, they serve the same purpose of speeding up training and improving predictions. These connections skip over the intermediate convolutions and connect the bottleneck layers. 
Shortcut connection is used to avoid vanishing gradients by enabling better gradient flow through the network during backpropagation.


#### The inverted residual block
The premise of the inverted residual layer is that: 
* Feature maps are able to be encoded in low-dimensional subspace 
* Non-linear activations result in information loss in site of their ability to increase representational complexity. 
This guide the design of the new convolutional layer.

The layer takes in a low-dimensional tensor and perform 3 separate convolutions:
1. 1x1 convolution (expansion layer): a point-wise convolution is used to expand the low-dimensional input feature map to a higher-dimensional space suited to non-linear activations, then ReLU6 is applied. The 1x1 convolution increases the number of channels. It is followed by the ReLU6 activation function which introduces non-linearity. It makes the depthwise convolutions more effective.
2. A depthwise convolution is performed followed by ReLU6 activation, achieving spatial convolution independently over each channel. This layer is also followed by ReLU6. We apply depthwise convolution after the expansion layer because the expanded set of features allows the depthwise convolution to process richer patterns across the expanded channels rather than just the original ones, improving the ability to detect meaningful features in the data.
3. A 1x1 convolution (projection layer): the partially-filtered feature map is projected back to a low-dimensional subspace using another point-wise convolution. This layer does not use an activation function, hence it is linear.


#### Architecture of MobileNetV2
Architecture of the MobileNetV2 is a follow:
1. Initial convolution layer: input layer and first convolution layer 
2. Serie of inverted residual blocks: each inverted residual block has a shortcut connection that skips over the depthwise convolution and connects directly from the input to the output, allowing for better gradient flow during training (this connection only exists when the input and output dimensions match). Each stage has a specific expansion factors, output channels, and strides to manage the computational complexity and receptive field
3. Convolutional layer: the final 1x1 convolution layer increases the channel dimension followed by a global average pooling layer
5. Fully connected layer: outputs the class scores for classification tasks (using a softmax when working with muticlass task)


### Fine-tuning
We fine-tune the later layers (can include not only the last layers but also some of the deeper feature extraction layers) of our model to capture high-level details near the end of the network and potentially improve accuracy.
These are the steps to fine-tune (in transfer learning):
1. Unfreeze the layers at the end of the network
2. Re-train our model on the final layers with a very low learning rate. A smaller learning rate allows us to take smaller steps to adapt the model a little more closely to our (new) data and it can lead to more fine details (and higher accuracy). We keep all other (first/previous) layers frozen.

We only fine-tune the later layers because in early layers, the model trains on low-level features (like edges), which is similar to many cases (similar features over images/cases). In later layers the model trains on more-complex and high-level features (like wispy hair or pointy ears). With new data (different than what the model got pre-trained on), we want the high-level features to adapt to it.


### Script description, step by step
* We load the packages needed for this project and we set up the right directory
* We create the training and validation datasets (from a directory)
* We preprocess and we augment data (using the Sequential API)
* We use a pretrained model (MobileNetV2) before realizing alpacas/not alpacas are not part of the model classes (so the model as it is is not performing)
* We adapt a pretrained model (MobileNetV2) to new data and train a classifier using the Functional API (weights being frozen) but the performance/accuracy is not that good
* We fine-tune a classifier's final layers to improve accuracy (from a pre-trained model). We unfreeze the weights at the rather end of the model to make them trainable on our data and get better performances


## References
This script is coming from the Deep Learning Specialization course. I enriched it to this new version.

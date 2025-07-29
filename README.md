# fashion-mnist-classifier
A convolutional neural network project made with PyTorch, to classify clothing items from the fashion MNIST dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Project Overview
This projects provides an end to end implementation of an image classifier for the fashion MNIST dataset. It is a CNN (Convolutional Neural Network) model designed to learn features from 28x28 greyscale images. The project code covers data loading, pre-processing, model building, training, and evaluation.

## Dataset

The Fashion-MNIST dataset is used for this project. It consists of:

    60,000 28x28 grayscale training images.

    10,000 28x28 grayscale testing images.

    10 Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

The dataset is loaded directly using the `keras.datasets.fashion_mnist.load_data()` function.

## Model Architecture

The model is a CNN built using PyTorch. It processes the image through two convolutional blocks before making a final classification. 

It involves the following layers

1. Convolutional Block 1: 32 Filters (2x2 Pool size)
2. Convolutional Block 2: 64 Filters (2x2 Pool size)
3. Classifier layer
     `Flatten` converts 2D feature maps to a 1D vector
     `nn.Linear` has 128 neurons, with ReLU activation
     `nn.Dropout` has a 50% rate to prevent overfitting
     `nn.Linear` (Output) has 10 neurons, one for each class

The model is trained using ADAM optimizer, and CrossEntropyLoss function.

## Results

The model achieves a final test accuracy of over 91%.

The `.ipynb` file contains a cell which has the code to plot the model's training loss and validation accuracy over time. 

The screenshot attached below shows the same

<img width="1485" height="606" alt="image" src="https://github.com/user-attachments/assets/8b528323-e30c-48be-a2f7-f1a63dec1ac0" />








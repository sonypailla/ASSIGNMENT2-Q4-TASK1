Task 1: Implementing AlexNet Architecture

Description

This task involves implementing a simplified version of the AlexNet architecture using TensorFlow/Keras. The model consists of convolutional layers, max-pooling layers, dense layers, dropout layers, and an output layer for classification.

Steps Implemented

Conv2D Layer with 96 filters, kernel size (11x11), stride 4, activation ReLU.

MaxPooling Layer with pool size (3x3), stride 2.

Conv2D Layer with 256 filters, kernel size (5x5), activation ReLU.

MaxPooling Layer with pool size (3x3), stride 2.

Three Conv2D Layers with 384, 384, and 256 filters, respectively.

MaxPooling Layer with pool size (3x3), stride 2.

Flatten Layer to convert the feature map into a dense layer input.

Fully Connected (Dense) Layers with 4096 neurons and ReLU activation.

Dropout Layers (50% dropout rate).

Output Layer with 10 neurons and Softmax activation.

Execution

Run the provided Python script to build and summarize the AlexNet model.

The model summary is printed, showing the number of parameters and layers.


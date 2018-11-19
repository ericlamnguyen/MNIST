# MNIST

Exploring the MNIST dataset.

1. Predicting with linear classifier with Average Perceptron algorithm

Command to run the program: python average_perceptron.py training_size epochs learning_rate
For example: python average_perceptron.py 10000 25 0.001

2. Predicting with linear classifier with logistic regression loss function and L2 regularization, the loss function is optimized with stochastic gradient descent method.

Command to run the program: python sgd.py [regularization? True / False] [Feature_type? type1 / type2] .

Feature_type? type1 / type2
+ type1 feature: each image is of size 28 x 28
+ type2 feature: the feature size is reduced by sampling the 28 x 28 original image into size of 14 x 14. Each block of 4 squares is represented by the highest value among them. 

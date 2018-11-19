import matplotlib
matplotlib.use('Agg')
import sys
import struct as st
import numpy as np
import os
import math
from matplotlib import pyplot as plt

#########################
#########################
def parseMNIST(dataset):
    """
    this function parses the binary data from MNIST files and return a numpy 2D
    array of the instances

    Params:
        dataset (str): "training" or "testing"
    Returns:
        data (numpy 2D array):
            if featue_type is "type1"
                A numpy 2D array with 786 columns, the first 784 columns
                are flattened 28x28 image matrix, the 785th column is
                the additional dimension to account for the bias term,
                the 786th column is the corresponding label of the instance

            if feature_type is "type2"
                A numpy 2D array with 198 columns, the first 196 columns
                are samples from type1's 784 columns, the 197th column is
                the additional dimension to account for the bias term,
                the 198th column is the corresponding label of the instance
    """

    if dataset is "training":
        img_file = 'train-images.idx3-ubyte'
        label_file = 'train-labels.idx1-ubyte'
    elif dataset is "testing":
        img_file = 't10k-images.idx3-ubyte'
        label_file = 't10k-labels.idx1-ubyte'

    with open(os.path.join(datapath, label_file), 'rb') as flbl:
        magic, num = st.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(os.path.join(datapath, img_file), 'rb') as fimg:
        magic, num, rows, cols = st.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    if feature_type == "type1":
        # an empty list
        data = []

        # flatten each 28x28 2D array into a 1D array, normalize the value and append to a single list
        for i in range(len(lbl)):
            row = np.reshape(img[i], 784)
            row = row * 1.0 / 255
            data.append(row)

        # convert data from list to 2D array
        data = np.array(data)

        # adding 1 more entry to each instance to account for bias term
        # bias_term is a 1D array with 1 column of 1s
        bias_term = np.ones((len(lbl), 1))

        # append additional column of 1s to data 2D array
        data = np.hstack((data, bias_term))

        # append additional column of corresponding labels to data 2D array
        lbl = np.reshape(lbl, (-1, 1))      # transpose the label array
        data = np.hstack((data, lbl))

        return data

    elif feature_type == "type2":
        # an empty list
        data = []

        for i in range(len(lbl)):
            # sample each image
            current_img = img[i]
            row = []        # sampled representation of each image
            for row_index in range(0, 27, 2):
                for col_index in range(0, 27, 2):
                    current_window = current_img[row_index:row_index + 1, col_index:col_index + 1]      # 2x2 window
                    max_of_window = np.amax(current_window)     # max value in the 2x2 window
                    row.append(max_of_window * 1.0 / 255)       # append the max value as a feature for the instance
            data.append(row)

        # convert data from list to 2D array
        data = np.array(data)

        # adding 1 more entry to each instance to account for bias term
        # bias_term is a 1D array with 1 column of 1s
        bias_term = np.ones((len(lbl),1))

        # append additional column of 1s to data 2D array
        data = np.hstack((data, bias_term))

        # append additional column of corresponding labels to data 2D array
        lbl = np.reshape(lbl, (-1, 1))
        data = np.hstack((data, lbl))

        return data


#########################################
#########################################
def F1_score(actual_labels, predictions):
    """
    this function calculate the macro F1 score

    Params:
        actual_labels (array)
        predictions (array)

    Returns:
        F1_score
    """

    len_ = len(actual_labels)
    F1s = []
    label_isPresent = [True for i in range(0,10)]

    # calculate F1 score and accuracy for each label
    for label in range(0,10):
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for j in range(0,len_):
            if predictions[j] == label:
                if actual_labels[j] == label:
                    TP += 1
                    continue
                else:
                    FP += 1
                    continue
            else:
                if actual_labels[j] == label:
                    FN += 1
                    continue
                else:
                    TN += 1
                    continue

        # calculate precision, recall, F1 score and accuracy for current label
        precision = TP*1.0 / (TP + FP + 0.0001)
        recall = TP*1.0 / (TP + FN + + 0.0001)

        # calculate F1 score for each label
        F1 = 2 * precision * recall / (precision + recall + 0.0001)
        F1s.append(F1)

        # determine if a label is present in the test set, if it is not it will
        # be ignored in the calculation of the final macro F1 score
        if TP == 0 and FN == 0 and FP == 0:
            label_isPresent[label] = False

    # calculate the final macro F1 score
    sum_F1 = 0
    len_F1 = 0
    for j in range(0,10):
        if label_isPresent[j] == True:
            sum_F1 += F1s[j]
            len_F1 += 1

    return (sum_F1 * 100.0 / len_F1)


###############################################
###############################################
def accuracy_score(actual_labels, predictions):
    """
    this function calculate the accuracy score

    Params:
        actual_labels (array)
        predictions (array)

    Returns:
        accuracy
    """

    correct_prediction = 0      # number of correct predictions
    total_predictions = len(predictions)

    for i in range(total_predictions):
        if actual_labels[i] == predictions[i]:
            correct_prediction += 1

    return correct_prediction * 1.0 / total_predictions


#######################
######################
def stochasticGD():
    """
    this function train the classifiers via Gradient Descent Algorithm

    Params:

    Returns:
        epochs: a list of epochs attempted through training
        training_losses: a list of training losses corresponding to each epoch
        training_accuracies: a list of training accuracies corresponding to each epoch
        test_accuracies: a list of test accuracies corresponding to each epoch
    """

    # initialize variables
    epoch = 0
    is_converged = False

    epochs = [0]
    training_losses = [1000]
    training_accuracies = [0]
    test_accuracies = [0]

    # generate 2D array, each row is a classifier whose index is the target label to predict
    Ws = np.random.uniform(low = -1.0, high = 1.0, size = (10, classifier_size))

    # loop until convergence
    while (not is_converged):
        epoch += 1
        epochs.append(epoch)

        np.random.shuffle(training_data)    # shuffle training data

        for i in range(1000):       # use 100 training examples per epoch
            x = training_data[i][0:classifier_size]     # training instance
            x_label = int(training_data[i][-1])         # training label

            Ys = [0,0,0,0,0,0,0,0,0,0]      # Ys[k] is the y value for the k_th classifier
            Ys[x_label] = 1                 # y = 1 for classifier whose index matches x_label, y = 0 for all other classifiers

            for j in range(10):     # for each of the classifier
                w = Ws[j]           # current classifer
                y = Ys[j]           # gold label for each classifier

                # calculate the gradient update
                z = np.dot(w,x)
                try:
                    e_z = math.exp(-1.0 * z)
                except OverflowError:
                    if z > 0:
                        e_z = 0.0
                    else:
                        e_z = 1000000.0

                g = 1.0 / (1.0 + e_z)

                delta_W = g * (1.0 - g) * (-y * 1.0 / (g + 1e-10) + (1 - y) * 1.0 / (1.0 - g + 1e-10)) * x

                # account for regularizer
                if regularization == "True":
                    L2_regularizer_gradient = LAMBDA * w

                    L2_regularizer_gradient[-1] = 0.0       # LAMBDA will not be applied to the bias term
                    delta_W += L2_regularizer_gradient

                # update the current classifier
                Ws[j] -= ALPHA * delta_W

        # compute the metrics with updated classifiers
        training_loss, training_accuracy, test_accuracy = computeMetrics(Ws)
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)

        # print out to console
        output = "epoch {0:d}: Training loss: {1:.2f}, Training Accuracy: {2:.2f}, Test Accuracy: {3:.2f}".format(epoch, training_loss, training_accuracy, test_accuracy)
        print(output)

        # stopping criterion
        if test_accuracies[-1] < test_accuracies[-2] and test_accuracies[-2] < test_accuracies[-3] and test_accuracies[-3] < test_accuracies[-4] or epoch == 200:
            is_converged = True

    # plot
    plt.plot(epochs[1:], training_accuracies[1:], 'b--', label = 'Train')
    plt.plot(epochs[1:], test_accuracies[1:], 'g', label = 'Test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((0.0, 1.0))   # set the ylim to bottom, top
    plt.savefig('convergence.png')


##########################
##########################
def computeMetrics(Ws):
    """
    this function computes the metrics to evaluate training convergence: training_loss, training_accuracy and test_accuracy

    params:
        Ws (2D float array): each row is a classifer
        classifier_size (int): the size of the classifier (full feature size or sampled feature size)

    returns:
        training_loss (float): value of the loss function given current classifier
        training_accuracy (float): accuracy of the classifiers on the training dataset
        test_accuracy (float): accuracy of the classifiers on the test dataset
    """

    #########################################
    # compute metrics on the training dataset
    actual_labels = []
    predictions = []
    data = training_data

    # loop through the training dataset
    training_loss = 0
    for i in range(len(data)):
        x = data[i, 0:classifier_size]
        x_label = int(data[i][-1])
        actual_labels.append(x_label)

        Ys = [0,0,0,0,0,0,0,0,0,0]      # Ys[k] is the y value for the k_th classifier
        Ys[x_label] = 1                 # y = 1 for classifier whose index matches x_label, y = 0 for all other classifiers

        # loop through the classifiers
        rank = []
        for j in range(10):
            w = Ws[j]       # current classifier
            y = Ys[j]           # gold label for each classifier

            z = np.dot(w,x)
            rank.append((z, j))     # index associating with highest z will be the prediction

            # compute the loss value
            try:
                e_z = math.exp(-1.0 * z)
            except OverflowError:
                if z > 0:
                    e_z = 0.0
                else:
                    e_z = 1000000.0

            g = 1.0 / (1.0 + e_z)
            training_loss += -y * math.log(g + 1e-10) - (1 - y) * math.log(1 - g + 1e-10)

        # select the winner
        rank = sorted(rank, reverse = True)
        prediction = rank[0][1]
        predictions.append(prediction)

    # calculate training accuracy
    training_accuracy = accuracy_score(actual_labels, predictions)

    # Calculate training losses
    if regularization == "True":
        training_loss_regularizer = 0
        for j in range(10):
            w = Ws[j]       # current classifier
            training_loss_regularizer += LAMBDA / 2 * np.sum(np.square(w[0:-1]))
        training_loss = training_loss * 1.0 / 100000  + training_loss_regularizer * 1.0 / 10
    else:
        training_loss = training_loss * 1.0 / 100000

    #####################################
    # compute metrics on the test dataset
    actual_labels = []
    predictions = []
    data = test_data

    # loop through the test dataset
    for i in range(len(data)):
        x = data[i, 0:classifier_size]
        x_label = int(data[i][-1])
        actual_labels.append(x_label)

        # loop through the classifiers
        rank = []
        for j in range(10):
            w = Ws[j]       # current classifier
            z = np.dot(w,x)
            rank.append((z, j))
        rank = sorted(rank, reverse = True)
        prediction = rank[0][1]
        predictions.append(prediction)

    test_accuracy = accuracy_score(actual_labels, predictions)

    return (training_loss, training_accuracy, test_accuracy)


##########################
##########################
if __name__ == "__main__":
    """
    main function
    """

    # read regularization setting from console
    regularization = (sys.argv[1])      # True / False
    if regularization != "True" and regularization != "False":
        print('ERROR: Unexpected input for regularization, only accepted "True" or "False"')
        print('')
        quit()

    # read feature_type setting from console
    feature_type = sys.argv[2]          # type1 / type2
    if feature_type == "type1":
        classifier_size = 785       # size of each classifier when type1 feature type (full feature size) is used, 785 = 784 + 1 (for bias term)
    elif feature_type == "type2":
        classifier_size = 197       # size of each classifier when type2 feature type (sampled feature size) is used, 197 = 196 + 1 (for bias term)
    else:
        print('ERROR: Unexpected input for feature_type, only accepted "type1" or "type2"')
        print('')
        quit()

    # read datapath from console
    datapath = sys.argv[3]

    # parse MNIST file and get the train and data set
    training_data = parseMNIST(dataset='training')[0:10000, :]      # only get the first 10,000 instances in the training set, discard the remaining 50,000
    test_data = parseMNIST(dataset='testing')

    # running batch gradient descent algorithm
    ALPHA = 0.01
    LAMBDA = 0.01
    stochasticGD()

import sys
import struct as st
import numpy as np
import gzip

def parseMNIST(dataset):
    """
    this function parses the binary data from MNIST files and return a numpy 2D
    array of the instances

    Params:
        dataset (str): "training" or "testing"
    Returns:
        data (numpy 2D array): A numpy 2D array with 786 columns, the first 784 columns
                            are flattened 28x28 image matrix, the 785th column is
                            the additional dimension to account for the bias term,
                            the 786th column is the corresponding label of the instance
    """

    if dataset is "training":
        img_file = 'train-images-idx3-ubyte.gz'
        label_file = 'train-labels-idx1-ubyte.gz'
    elif dataset is "testing":
        img_file = 't10k-images-idx3-ubyte.gz'
        label_file = 't10k-labels-idx1-ubyte.gz'

    with gzip.open(label_file, 'rb') as flbl:
        magic, num = st.unpack(">II", flbl.read(8))
        lbl = np.fromstring(flbl.read(), dtype=np.int8)

    with gzip.open(img_file, 'rb') as fimg:
        zero, data_type, dims = st.unpack('>HBB', fimg.read(4))
        shape = tuple(st.unpack('>I', fimg.read(4))[0] for d in range(dims))
        img = np.fromstring(fimg.read(), dtype=np.uint8).reshape(shape)

    # an empty list
    data = []

    # flatten each 28x28 2D array into a 1D array, normalize the value and append to data
    for i in range(len(lbl)):
        row = img[i].reshape(784)
        row = np.around((row * 1.0 / 255))
        data.append(row)

    # convert data from list to 2D array
    data = np.array(data)

    # adding 1 more entry to each instance to account for bias term
    # bias_term is a 1D array with 1 column of 1s
    bias_term = np.ones((len(lbl),1))

    # append additional column of 1s to data 2D array
    data = np.hstack((data, bias_term))

    # append additional column of corresponding labels to data 2D array
    lbl = lbl.reshape(-1, 1)
    data = np.hstack((data, lbl))

    return data


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
        if TP == 0 and FN == 0:
            label_isPresent[label] = False

    # calculate the final macro F1 score
    sum_F1 = 0
    len_F1 = 0
    for j in range(0,10):
        if label_isPresent[j] == True:
            sum_F1 += F1s[j]
            len_F1 += 1

    return (sum_F1 * 100.0 / len_F1)


def train_average_perceptron(training_data, training_size, epoch, r):
    """
    this function train the classifiers via Vanilla Perceptron Algorithm

    Params:
        training_data (numpy 2D array): 10,000 training instances
        training_size (int): Number of training instances used
        epoch (int): Number of iterations to run through the training instances
        r (float): learning rate

    Returns:
        Ws (numpy 2D array): 10 classifiers w corresponding to 10 labels from 0 to 9
    """

    # generate 2D array, each row is an array of 785 randomly generated values
    Ws = np.random.rand(10, 785)
    Ws_sum = np.zeros((10, 785), dtype=float)

    # loop epoch iterations
    for i in range(epoch):
        # shuffle training data
        np.random.shuffle(training_data)
        # loop through the training instances
        for j in range(training_size):
            x = training_data[j, 0:785]
            label = training_data[j, 785]
            # loop through each of the classifiers
            for k in range(10):
                w = Ws[k]
                if np.dot(x,w) >= 0:
                    y_prediction = 1
                else:
                    y_prediction = 0

                # if the label is the same as the index of the classifer, actual_label = 1
                # else it is 0
                y_actual = 0
                if k == label:
                    y_actual = 1

                # update the classifier w and Ws
                w = w + (y_actual - y_prediction) * 1.0 * r * x
                Ws[k] = w
                Ws_sum[k] += w

    # average the identifiers
    count = epoch * training_size

    return Ws_sum * 1.0 / count


def predict_average_perceptron(Ws, data):
    """
    this function use the learned classifiers to predict the label

    Params:
        Ws (numpy 2D array): Each row is a learned classifier, the index of the
                            classifier is the target label that it is supposed
                            to predict
        data (numpy 2D array): The dataset to perform prediction on

    Returns:
        Macro F1 score for the classifiers on the dataset
    """

    actual_labels = []
    predictions = []

    # loop through the dataset
    for i in range(len(data)):
        x = data[i, 0:785]
        label = data[i, 785]
        actual_labels.append(label)
        # loop through the classifiers
        rank = []
        for j in range(10):
            w = Ws[j]
            xTw = np.dot(x,w)
            rank.append((xTw, j))
        rank = sorted(rank, reverse = True)
        prediction = rank[0][1]
        predictions.append(prediction)

    return F1_score(actual_labels, predictions)


if __name__ == "__main__":
    """
    main function
    """

    # read arguments from console
    training_size = int(sys.argv[1])
    epoch = int(sys.argv[2])
    r = float(sys.argv[3])

    # parse MNIST file and get the train and data set
    # only get the first 10,000 instances in the training set, discard the remaining 50,000
    training_data = parseMNIST(dataset='training')[0:10000, :]
    test_data = parseMNIST(dataset='testing')

    # train the classifers via Vanilla Perceptron Algorithm
    Ws = train_average_perceptron(training_data, training_size, epoch, r)
    # calculate F1 scores for training data and test data
    F1_score_training_data = predict_average_perceptron(Ws, training_data)
    F1_score_test_data = predict_average_perceptron(Ws, test_data)

    # print out to console
    print('')
    print("Training F1 score: " + str(round(F1_score_training_data / 100.0, 2)))
    print("Test F1 score: " + str(round(F1_score_test_data / 100.0, 2)))
    print('')

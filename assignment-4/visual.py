import tensorflow as tf
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt

# Hyper parameters
HIDDEN_LAYER_NODES_OPTIONS = [300]
DROPOUT_RATE_OPTIONS = [1.0]
LEARNING_RATE = 0.001
BATCH_SIZE = 5000
EPOCHS = 100

# Changed internally, do not change
DROPOUT_RATE = 1.0
HIDDEN_LAYER_NODES = 100

SAVE_PATH = "./checkpoint-reg/save/model.ckpt"
SAVE_FILE = "./predictions-reg/submission.csv"

mins = [0] * 91
ptps = [0] * 91

DEGREE = 1
MULTIPLY_FEATURES = False
INPUT_NODES = 90


def checkArgs():
    if (len(sys.argv) != 4):
        print "Please enter three arguments. For instance, run: \
        \npython visual.py train_reg.csv dev_reg.csv test-reg.csv"
        exit(0)

    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    test_file = sys.argv[3]

    return train_file, dev_file, test_file


def readCSV(csv_file):
    # open the training file file in universal line ending mode
    # Courtesy: https://stackoverflow.com/a/29082892
    with open(csv_file, 'rU') as infile:
        # read the file as a dictionary for each row ({header : value})
        reader = csv.DictReader(infile)
        parameters = reader.fieldnames
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]
    return data, parameters


def plotGraph(X, Y, p):
    plt.scatter(X, Y)
    plt.title(p + ' vs year')
    plt.xlabel(p)
    plt.ylabel('year')
    plt.show()


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = np.asarray([labels[i] for i in idx])

    return np.asarray(data_shuffle), np.reshape(labels_shuffle, (-1, 1))


if __name__ == "__main__":
    # Check args
    train_file, dev_file, test_file = checkArgs()

    num_features = 90

    ########################
    # Read training set data
    train_data, parameters = readCSV(train_file)
    train_examples = [[0]] * (len(parameters))
    train_labels = np.array([float(x) for x in train_data[parameters[0]]])
    for i in xrange(1, len(parameters)):
        train_examples[i] = [float(x) for x in train_data[parameters[i]]]
        mins[i], ptps[i] = np.min(train_examples[i]), np.ptp(train_examples[i])
        train_examples[i] = [(float(x) - mins[i]) / ptps[i]
                             for x in train_data[parameters[i]]]

    if (MULTIPLY_FEATURES):
        for a in xrange(1, num_features + 1):
            for b in xrange(a + 1, num_features + 1):
                train_examples.append(np.multiply(
                    train_examples[a], train_examples[b]))

    for a in xrange(1, num_features + 1):
        for d in xrange(2, DEGREE + 1):
            train_examples.append(np.array(train_examples[a]) ** d)

    train_examples = np.array(train_examples[1:]).transpose()
    print("Completed reading training data")

    for i in xrange(0, len(parameters)):
        X = [float(x) for x in train_data[parameters[i]]]
        plotGraph(X, train_labels, parameters[i])
        print np.corrcoef(train_labels, X)

    ###################
    # Read dev set data
    dev_data, parameters = readCSV(dev_file)
    test_examples = [[0]] * (len(parameters))
    test_labels = np.array([float(x) for x in dev_data[parameters[0]]])
    for i in xrange(1, len(parameters)):
        test_examples[i] = [(float(x) - mins[i]) / ptps[i]
                            for x in dev_data[parameters[i]]]

    if (MULTIPLY_FEATURES):
        for a in xrange(1, num_features + 1):
            for b in xrange(a + 1, num_features + 1):
                test_examples.append(np.multiply(
                    test_examples[a], test_examples[b]))

    for a in xrange(1, num_features + 1):
        for d in xrange(2, DEGREE + 1):
            test_examples.append(np.array(test_examples[a]) ** d)

    test_examples = np.array(test_examples[1:]).transpose()
    print("Completed reading dev data")

    ###################
    # Read dev set data
    test_data, parameters = readCSV(test_file)
    test_samples = [[0]] * (len(parameters))
    ids = np.array([float(x) for x in test_data[parameters[0]]]).astype(int)
    for i in xrange(1, len(parameters)):
        test_samples[i] = [(float(x) - mins[i]) / ptps[i]
                           for x in test_data[parameters[i]]]

    if (MULTIPLY_FEATURES):
        for a in xrange(1, num_features + 1):
            for b in xrange(a + 1, num_features + 1):
                test_samples.append(np.multiply(
                    test_samples[a], test_samples[b]))

    for a in xrange(1, num_features + 1):
        for d in xrange(2, DEGREE + 1):
            test_samples.append(np.array(test_samples[a]) ** d)

    INPUT_NODES = len(test_samples) - 1

    test_samples = np.array(test_samples[1:]).transpose()
    print("Completed reading test data")

#!/usr/bin/python

import csv
import sys
import numpy as np
import time


def checkArgs():
    """Checks if the required arguments are specified while execution."""

    if (len(sys.argv) != 4):
        print "Please enter three arguments. For instance, run: \
        \npython lr.py train.csv test.csv 0.005"
        exit(0)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    try:
        _eta = float(sys.argv[3])
    except ValueError:
        print "Please enter a float for the eta argument"

    return train_file, test_file, _eta


def readCSV(csv_file):
    """
    Reads a CSV file and extracts data.extracts
    Courtesy: https://stackoverflow.com/a/29082892

    Arguments-
    csv_file: path of CSV file to be read

    Returns-
    data: Dict of extracted data from the file
    parameters: list of all fieldnames
    """

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


def gradientDescent(X, Y, W, N, _eta, maxIterations=1000000):
    """
    Implements the batch gradient descent algorithm.

    Arguments-
    X: input matrix for training set
    Y: actual output for training set

    """
    prev_cost = 0.0
    start_time = time.time()
    for i in xrange(0, maxIterations):
        loss = [0.0] * N
        F = np.dot(X, W)
        loss = F - Y

        cost = np.sum(loss ** 2) / (2 * N)
        elapsed = (time.time() - start_time) / 60
        remaining = (maxIterations - (i + 1)) * elapsed / (i + 1)
        print("Iteration % d | Cost: % f | Elapsed: % 0.1f min | Remaining: % 0.1f min" % (
            i, cost, elapsed, remaining))

        # if (np.abs(prev_cost - cost) < 0.00000000000000001):
        #     break
        # prev_cost = cost

        gradient = np.dot(X.transpose(), loss) / N
        W = W - _eta * gradient

    return W


if __name__ == '__main__':
    train_file, test_file, _eta = checkArgs()

    train_data, parameters = readCSV(train_file)

    parameters.remove('price')

    train_limit = 20000

    # Prepare actual output vector.
    # Dimensions Nx1
    Y = [float(x) for x in train_data['price']]
    Y_min = np.min(Y)
    Y_ptp = np.ptp(Y)
    Y = [(float(x) - Y_min) / Y_ptp for x in train_data['price']]
    Y = np.array(Y)
    Y = Y[:train_limit]

    # Number of data points
    N = len(Y)

    # Number of input parameters
    k = 1

    # Prepare input matrix X.
    mins = [0] * (k + 1)
    ptps = [1] * (k + 1)
    X = [[0]] * (k + 1)

    # Setting x_i_1 to 1 for all x_i
    X[0] = [1] * N
    X[1] = [float(x) for x in train_data['sqft_lot']]
    mins[1] = np.min(X[1])
    ptps[1] = np.ptp(X[1])
    X[1] = [(float(x) - mins[1]) / ptps[1]
            for x in train_data['sqft_lot']]

    X[0] = X[0][:train_limit]
    X[1] = X[1][:train_limit]

    total_features = len(X)
    print total_features

    # Dimensions Nx(k+1)
    X = np.array(X).transpose()

    # Initialising linear predictors, W
    # Dimensions (k+1)x1
    W = [0.0] * (total_features)
    W = np.array(W)

    W = gradientDescent(X, Y, W, N, _eta)

    test_data, para = readCSV(test_file)
    price = []
    for i in xrange(0, len(test_data['id'])):
        Q = [0] * (k + 1)
        Q[0] = 1

        Q[1] = float(test_data['sqft_lot'][i])
        Q[1] = (Q[1] - mins[1]) / ptps[1]

        Q = np.array(Q)
        p = np.dot(Q, W)
        price.append(p * Y_ptp + Y_min)

    ids = [int(x) for x in test_data['id']]
    out = np.asarray([ids, price])
    np.savetxt("submission.csv", out.transpose(),
               '%d', delimiter=",", header="id,price", comments='')

#!/usr/bin/python

import csv
import sys
import numpy as np
import time
import signal


def checkArgs():
    """Checks if the required arguments are specified while execution."""

    if (len(sys.argv) != 5):
        print "Please enter four arguments. For instance, run: \
        \npython lr.py train.csv test.csv 0.005 5"
        exit(0)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    try:
        _eta = float(sys.argv[3])
    except ValueError:
        print "Please enter a float for the eta argument"
    try:
        lamb = float(sys.argv[4])
    except ValueError:
        print "Please enter a float for the lambda argument"

    return train_file, test_file, _eta, lamb


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


def gradientDescent(X, Y, W, N, _eta, lamb, degree=1, maxIterations=1000000):

    def save_W():
        print "Saving linear predictors"
        print W
        np.savetxt('W.txt', W, fmt='%5.15f')

    def exit_gracefully(signal, frame):
        save_W()
        sys.exit(0)

    signal.signal(signal.SIGINT, exit_gracefully)
    start_time = time.time()
    prev_W = np.copy(W)
    prev_cost = 999
    for i in xrange(0, maxIterations):
        loss = [0.0] * N
        F = np.dot(X, W)
        loss = F - Y

        cost = (np.sum(loss ** 2) + lamb * np.sum(W ** 2)) / (2 * N)
        elapsed = (time.time() - start_time) / 60
        remaining = (maxIterations - (i + 1)) * elapsed / (i + 1)
        print("Iteration % d | Cost: % f | LR: % 0.3f | Elapsed: % 0.1f min | Remaining: % 0.1f min" % (
            i, cost, _eta, elapsed, remaining))

        # if (np.abs(prev_cost - cost) < 0.00000000000000001):
        #     break

        if (cost - prev_cost > 0):
            print("OVERSHOOT Cost: %0.10f | Prev Cost: %0.10f" %
                  (cost, prev_cost))
            W = np.copy(prev_W)
            _eta = _eta / 1.2
        else:
            if (i % 100 == 0):
                _eta = _eta * 1.2
            prev_cost = cost
            prev_W = np.copy(W)
            gradient = (np.dot(X.transpose(), loss) + lamb * W) / N
            W = W - _eta * gradient

    save_W()
    return W


if __name__ == '__main__':
    train_file, test_file, _eta, lamb = checkArgs()

    train_data, parameters = readCSV(train_file)
    parameters.remove('price')
    # id,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,
    # waterfront,view,condition,grade,sqft_above,sqft_basement,
    # yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15,date
    worthless = ['date']
    for item in worthless:
        parameters.remove(item)

    # Prepare actual output vector.
    # Dimensions Nx1
    Y = [float(x) for x in train_data['price']]
    Y_min = np.min(Y)
    Y_ptp = np.ptp(Y)
    Y = [(float(x) - Y_min) / Y_ptp for x in train_data['price']]
    Y = np.array(Y)

    # Number of data points
    N = len(Y)

    # Number of input parameters
    k = 19 - len(worthless)

    # Degree
    degree = 8

    # Prepare input matrix X.
    mins = [0] * (k + 1)
    ptps = [1] * (k + 1)
    X = [[0]] * (k + 1)

    # Setting x_i_1 to 1 for all x_i
    X[0] = [1] * N
    for i in xrange(1, k + 1):
        X[i] = [float(x) for x in train_data[parameters[i]]]
        mins[i] = np.min(X[i])
        ptps[i] = np.ptp(X[i])
        X[i] = [(float(x) - mins[i]) / ptps[i]
                for x in train_data[parameters[i]]]

    for a in xrange(1, k + 1):
        for b in xrange(a + 1, k + 1):
            X.append(np.multiply(X[a], X[b]))

    for a in xrange(1, len(X) + 1):
        for d in xrange(2, degree + 1):
            X.append(np.array(X[a]) ** d)

    total_features = len(X)
    print total_features

    # Dimensions Nx(k+1)
    X = np.array(X).transpose()
    print np.shape(X)

    # Initialising linear predictors, W
    # Dimensions (k+1)x1
    W = [0.0] * (total_features)
    W = np.array(W)

    try:
        W = np.loadtxt('W.txt', dtype=np.float32)
        W = np.array(W)
        print "Reloading saved linear predictors"
    except IOError:
        print "Initialising new linear predictors"

    W = gradientDescent(X, Y, W, N, _eta, lamb, degree)

    test_data, para = readCSV(test_file)
    price = []
    for i in xrange(0, len(test_data['id'])):
        Q = [0] * (k + 1)
        Q[0] = 1

        for j in xrange(1, k + 1):
            Q[j] = float(test_data[parameters[j]][i])
            Q[j] = (Q[j] - mins[j]) / ptps[j]

        for a in xrange(1, k + 1):
            for b in xrange(a + 1, k + 1):
                Q.append(np.multiply(Q[a], Q[b]))

        for a in xrange(1, len(Q) + 1):
            for d in xrange(2, degree + 1):
                Q.append(Q[a] ** d)

        Q = np.array(Q)
        p = np.dot(Q, W)
        price.append(p * Y_ptp + Y_min)

    ids = [int(x) for x in test_data['id']]
    out = np.asarray([ids, price])
    np.savetxt("submission.csv", out.transpose(),
               '%d', delimiter=",", header="id,price", comments='')

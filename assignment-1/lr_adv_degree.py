import csv
import sys
import numpy as np


def checkArgs():
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


def gradientDescent(X, W, N, degree=1, maxIterations=1000000):
    prev_cost = 0.0
    for i in xrange(0, maxIterations):
        loss = [0.0] * N
        F = [[0.0] * N] * degree
        for j in xrange(0, degree):
            F[j] = np.dot(X[j], W[j])
            loss = loss + F[j]
        loss = loss - Y

        cost = np.sum(loss ** 2) / (2 * N)
        print("Iteration %d | Cost: %f" % (i, cost))

        if (np.abs(prev_cost - cost) < 0.00000000000000001):
            break
        prev_cost = cost

        for j in xrange(0, degree):
            gradient = np.dot(X[j].transpose(), loss) / N
            W[j] = W[j] - _eta * gradient

    return W


if __name__ == '__main__':
    train_file, test_file, _eta = checkArgs()

    train_data, parameters = readCSV(train_file)
    parameters.remove('price')

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
    k = 19

    # Degree
    degree = 10

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

    # Dimensions Nx(k+1)
    X = np.array(X).transpose()
    X = [X**x for x in xrange(1, degree + 1)]
    X = np.array(X)

    # Initialising linear predictors, W
    # Dimensions (k+1)x1
    W = [0.0] * (k + 1)
    W = np.array(W)
    W = [W] * degree
    W = np.array(W)

    W = gradientDescent(X, W, N, degree)

    test_data, parameters = readCSV(test_file)
    price = []
    for i in xrange(0, len(test_data['id'])):
        Q = [0] * (k + 1)
        Q[0] = 1
        for j in xrange(1, k + 1):
            Q[j] = float(test_data[parameters[j]][i])
            Q[j] = (Q[j] - mins[j]) / ptps[j]
        Q = np.array(Q)
        Q = [Q**x for x in xrange(1, degree + 1)]
        X = np.array(X)
        p = 0.0
        for j in xrange(0, degree):
            p += np.dot(Q[j], W[j])
        price.append(p * Y_ptp + Y_min)

    ids = [int(x) for x in test_data['id']]
    out = np.asarray([ids, price])
    np.savetxt("submission.csv", out.transpose(),
               '%d', delimiter=",", header="id,price", comments='')

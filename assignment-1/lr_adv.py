import csv
import sys
import numpy as np
import time
import signal


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


def gradientDescent(X, W, N, _eta, degree=1, maxIterations=400000):
    prev_cost = 0.0
    start_time = time.time()
    prev_W = W
    prev_cost = 999
    F = [[0.0] * N] * degree

    for i in xrange(0, maxIterations):
        loss = [0.0] * N
        for j in xrange(0, degree):
            F[j] = np.dot(X[j], W[j])
            loss = loss + F[j]
        loss = loss - Y

        cost = np.sum(loss ** 2) / (2 * N)
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
            for j in xrange(0, degree):
                gradient = np.dot(X[j].transpose(), loss) / N
                W[j] = W[j] - _eta * gradient

    return W


def exit_gracefully(signal, frame):
    print "Saving linear predictors"
    np.savetxt('W.txt', W)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, exit_gracefully)
    train_file, test_file, _eta = checkArgs()

    train_data, parameters = readCSV(train_file)
    parameters.remove('price')
    # id,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,
    # waterfront,view,condition,grade,sqft_above,sqft_basement,
    # yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15,date
    worthless = []
    for item in worthless:
        parameters.remove(item)

    for i in xrange(0, len(train_data['yr_renovated'])):
        if int(train_data['yr_renovated'][i]) > 0:
            train_data['yr_renovated'][i] = 1

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
    degree = 1

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
    total_features = len(X)
    print total_features

    # Dimensions Nx(k+1)
    X = np.array(X).transpose()
    log_X = 0.001 * np.ma.log(np.abs(X)).filled(0)
    X = [X**x for x in xrange(1, degree + 1)]
    X.append(log_X)
    X = np.array(X)

    # Initialising linear predictors, W
    # Dimensions (k+1)x1
    W = [0.0] * (total_features)
    W = np.array(W)
    W = [W] * (degree + 1)  # + 1 for log
    W = np.array(W)
    try:
        W = np.loadtxt('W.txt', dtype=np.float64)
        W = np.array(W)
        print "Reloading saved linear predictors"
    except IOError:
        print "Initialising new linear predictors"

    W = gradientDescent(X, W, N, _eta, degree + 1)

    test_data, para = readCSV(test_file)
    for i in xrange(0, len(test_data['yr_renovated'])):
        if int(test_data['yr_renovated'][i]) > 0:
            test_data['yr_renovated'][i] = 1
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

        Q = np.array(Q)
        log_Q = 0.001 * np.ma.log(np.abs(Q)).filled(0)
        Q = [Q**x for x in xrange(1, degree + 1)]
        Q.append(log_Q)
        X = np.array(X)
        p = 0.0
        for j in xrange(0, degree + 1):
            p += np.dot(Q[j], W[j])
        price.append(p * Y_ptp + Y_min)

    ids = [int(x) for x in test_data['id']]
    out = np.asarray([ids, price])
    np.savetxt("submission.csv", out.transpose(),
               '%d', delimiter=",", header="id,price", comments='')

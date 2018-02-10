import csv
import sys
import numpy as np
import matplotlib.pyplot as plt


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


def gradientDescent(X, W, maxIterations=100000):
    prev_cost = 0.0
    for i in xrange(0, maxIterations):
        F = np.dot(X, W)
        loss = F - Y

        cost = np.sum(loss ** 2) / (2 * N)
        # print("Iteration %d | Cost: %f" % (i, cost))

        if (np.abs(prev_cost - cost) < 0.00000000000001):
            break
        prev_cost = cost

        gradient = np.dot(X.transpose(), loss) / N
        W = W - _eta * gradient

    return W


def plotGraph(X, Y, W):
    x_0 = 0
    y_0 = W[0]

    x_1 = 16
    y_1 = W[1] * (x_1 - x_0) + W[0]

    plt.scatter(np.array(X).transpose()[1], Y)
    plt.title('sqft_lot vs price')
    plt.xlabel('sqft_lot (x10^5)')
    plt.ylabel('price (x10^5)')
    # Draw these two points with big triangles to make it clear
    # where they lie
    plt.scatter([x_0, x_1], [y_0, y_1], marker='.', s=150, c='r')

    # And now connect them
    plt.plot([x_0, x_1], [y_0, y_1], c='r')
    plt.show()

    # # Find the slope and intercept of the best fit line
    # slope, intercept = np.polyfit(np.array(X).transpose()[1], Y, 1)
    # print slope, intercept
    # # Create a list of values in the best fit line
    # abline_values = [slope * i + intercept for i /
    #                  in np.array(X).transpose()[1]]

    # # Plot the best fit line over the actual values
    # plt.plot(np.array(X).transpose()[1], Y, '.')
    # plt.plot(np.array(X).transpose()[1], abline_values, 'b')
    # plt.title(slope)
    # plt.show()


if __name__ == '__main__':
    train_file, test_file, _eta = checkArgs()

    train_data, parameters = readCSV(train_file)

    # Prepare actual output vector.
    # Dimensions Nx1
    Y = [float(x) / 100000 for x in train_data['price']]
    Y = np.array(Y)

    # Number of data points
    N = len(Y)

    # Number of input parameters
    k = 1

    # Prepare input matrix X.
    X = [[0]] * (k + 1)
    # Setting x_i_1 to 1 for all x_i
    X[0] = [1] * N
    # for i in xrange(1, k + 1):
    #     X[i] = ...
    X[1] = [float(x) / 100000 for x in train_data['sqft_lot']]

    # Dimensions Nx(k+1)
    X = np.array(X).transpose()

    # Initialising linear predictors, w = [w1, w2]
    # Hypothesis- f(x) = w1 + w2*x
    # Dimensions (k+1)x1
    W = [0] * (k + 1)
    W = np.array(W)

    W = gradientDescent(X, W)
    print W

    plotGraph(X, Y, W)

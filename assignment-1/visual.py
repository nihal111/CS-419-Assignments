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
        print "Please enter a float for the enterta argument"

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


def plotGraph(X, Y, p):
    plt.scatter(X, Y)
    plt.title(p + ' vs price')
    plt.xlabel(p)
    plt.ylabel('price')
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
    Y = [float(x) for x in train_data['price']]
    Y = np.array(Y)

    # Number of data points
    N = len(Y)

    # Number of input parameters
    k = 1

    for i in xrange(0, len(parameters)):
        X = [float(x) for x in train_data[parameters[i]]]
        plotGraph(X, Y, parameters[i])
        print np.corrcoef(Y, X)

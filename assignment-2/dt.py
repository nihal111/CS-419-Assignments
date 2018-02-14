import csv
import sys
import numpy as np
from math import log


class Node:
    """A node in the decision tree"""

    def __init__(self, attribute="Unset"):
        self.attribute = attribute
        self.children = []
        self.answer = None
        self.divide = None  # For continuous distribution only

    def __str__(self):
        return "Name- " + self.attribute + " Answer- " + str(self.answer)


def checkArgs():
    if (len(sys.argv) != 4):
        print "Please enter three arguments. For instance, run: \
        \npython dt.py train.csv test.csv 5"
        exit(0)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    try:
        depth = int(sys.argv[3])
    except ValueError:
        print "Please enter an integer for the depth argument"

    return train_file, test_file, depth


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


def entropy(examples, target_attribute, attributes):
    """Calculates the entropy of the given data set for the target attr"""

    frequency = {}
    data_entropy = 0.0

    # find index of the target attribute
    i = attributes.index(target_attribute)

    # Calculate the frequency of each of the values in the target_attribute
    for entry in examples:
        if (entry[i] in frequency):
            frequency[entry[i]] += 1.0
        else:
            frequency[entry[i]] = 1.0

    # Calculate the entropy of the data for the target attr
    for freq in frequency.values():
        data_entropy += (-freq / len(examples)) * \
            log(freq / len(examples), 2)

    return data_entropy


def gain_discrete(examples, target_attribute, attributes, attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    frequency = {}
    subset_entropy = 0.0

    # find index of the chosen attr
    i = attributes.index(attr)

    # Calculate the frequency of each of the values in the chosen attr
    for entry in examples:
        if (entry[i] in frequency):
            frequency[entry[i]] += 1.0
        else:
            frequency[entry[i]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in frequency.keys():
        val_prob = frequency[val] / sum(frequency.values())
        data_subset = [entry for entry in examples if entry[i] == val]
        subset_entropy += val_prob * \
            entropy(data_subset, target_attribute, attributes)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(examples, target_attribute, attributes) - subset_entropy)


def gain_continuous(examples, target_attribute, attributes, attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """

    # find index of the chosen attr
    i = attributes.index(attr)
    value_list = np.sort(np.array(examples).transpose()[i])
    min_value = value_list[0]
    max_value = value_list[-1]
    range_value = max_value - min_value
    STEP_SIZE = 20  # in percent
    divide = min_value + STEP_SIZE * range_value / 100
    best_divide = min_value
    best_subset_entropy = 0.0

    print value_list

    while (divide < max_value - STEP_SIZE * range_value / 100):
        less_subset = []
        more_subset = []
        for entry in examples:
            if (entry[i] < divide):
                less_subset.append(entry)
            else:
                more_subset.append(entry)
        subset_entropy = 0.0
        subset_entropy += len(less_subset) * \
            entropy(less_subset, target_attribute, attributes) / len(examples)
        subset_entropy += len(more_subset) * \
            entropy(more_subset, target_attribute, attributes) / len(examples)

        if (best_subset_entropy < subset_entropy):
            best_subset_entropy = subset_entropy
            best_divide = divide

        divide += STEP_SIZE * range_value / 100

    print "best_divide = " + str(best_divide)
    return (entropy(examples, target_attribute, attributes) -
            best_subset_entropy), best_divide


def find_best_attribute(examples, target_attribute, attributes):
    best_attribute = attributes[0]
    max_gain = 0
    best_divide = None
    for a in attributes:
        if a == target_attribute or a == "id":
            continue
        print "Checking for attribute- " + a

        attr_index = attributes.index(a)
        options = np.unique(np.array(examples).transpose()
                            [attr_index]).shape[0]
        if options < 50:
            newGain = gain_discrete(examples, target_attribute, attributes, a)
            divide = None  # Only for continuos
        else:
            newGain, divide = gain_continuous(
                examples, target_attribute, attributes, a)
        print "Gain- " + str(newGain)
        if newGain > max_gain:
            max_gain = newGain
            best_attribute = a
            best_divide = divide
    return best_attribute, best_divide


def ID3(examples, target_attribute, attributes):

    # Create root node
    root = Node()
    print attributes
    price_index = 1

    # Return single node, if all examples have same class
    # Find list of unique values in target_attribute of examples
    target_list = np.unique(np.array(examples).transpose()[price_index])

    if target_list.shape[0] == 1:
        root.answer = target_list[0]
        return root

    # If attributes is empty, return tree with most common value
    if len(attributes) == 0:
        root.answer = np.argmax(np.bincount(
            np.array(examples).transpose()[price_index]))
        return root

    # Main algo
    best_attribute, best_divide = find_best_attribute(
        examples, target_attribute, attributes)

    print best_attribute

    root.attribute = best_attribute
    root.divide = best_divide

    attr_index = attributes.index(best_attribute)
    options = np.unique(np.array(examples).transpose()[attr_index])
    for value in options:
        branch = Node()
        example_subset = [
            entry for entry in examples if entry[attr_index] == value]
        branch = ID3(example_subset, target_attribute, list(
            set(attributes) - set([best_attribute])))
        root.children.append(branch)

    # Add a default node, for values outside options available in training data
    branch = Node()
    branch.answer = np.argmax(np.bincount(
        np.array(examples).transpose()[price_index]))
    root.children.append(branch)

    return root


if __name__ == '__main__':
    train_file, test_file, depth = checkArgs()

    train_data, parameters = readCSV(train_file)

    examples = [[0]] * len(parameters)

    for i in xrange(0, len(parameters)):
        # examples[i] = [float(x) for x in train_data[parameters[i]]]
        if (parameters[i] == "floors"):
            examples[i] = [int(2 * float(x))
                           for x in train_data[parameters[i]]]
        else:
            examples[i] = [int(x) for x in train_data[parameters[i]]]

    examples = np.array(examples).transpose()

    tree = ID3(examples, "price", parameters)

    print tree

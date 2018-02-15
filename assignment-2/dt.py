import csv
import sys
import numpy as np
from math import log


class Node:
    """A node in the decision tree"""

    def __init__(self):
        self.attribute = "Leaf"  # By default a leaf, unless attr specified
        self.children = []
        self.condition = None  # For every child node
        self.answer = None  # Only for leaf nodes, answer for the filter
        self.divide = None  # For continuous distribution only

    def __str__(self):
        s = "Name- " + self.attribute + " Answer- " + str(self.answer) + \
            " Condition-" + str(self.condition) + " Divide-" + str(self.divide)
        return s


def checkArgs():
    if (len(sys.argv) != 4):
        print "Please enter three arguments. For instance, run: \
        \npython dt.py train.csv test.csv 5"
        exit(0)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    try:
        MAX_DEPTH = int(sys.argv[3])
    except ValueError:
        print "Please enter an integer for the depth argument"

    return train_file, test_file, MAX_DEPTH


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


def entropy(examples, target_attribute, attributes, all_parameters):
    """Calculates the entropy of the given data set for the target attr"""

    frequency = {}
    data_entropy = 0.0

    # find index of the target attribute
    i = all_parameters.index(target_attribute)

    # Calculate the frequency of each of the values in the target_attribute
    for example in examples:
        if (example[i] in frequency):
            frequency[example[i]] += 1.0
        else:
            frequency[example[i]] = 1.0

    # Calculate the entropy of the data for the target attr
    for freq in frequency.values():
        data_entropy += (-freq / len(examples)) * \
            log(freq / len(examples), 2)

    return data_entropy


def gain_discrete(examples, target_attribute,
                  attributes, attr, all_parameters):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    frequency = {}
    subset_entropy = 0.0

    # find index of the chosen attr
    i = all_parameters.index(attr)

    # Calculate the frequency of each of the values in the chosen attr
    for example in examples:
        if (example[i] in frequency):
            frequency[example[i]] += 1.0
        else:
            frequency[example[i]] = 1.0

    # Calculate the weighted sum of entropies in each subset of values
    for val in frequency.keys():
        val_prob = frequency[val] / sum(frequency.values())
        data_subset = [example for example in examples if example[i] == val]
        subset_entropy += val_prob * \
            entropy(data_subset, target_attribute, attributes, all_parameters)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(examples,
                    target_attribute,
                    attributes,
                    all_parameters
                    ) - subset_entropy)


def gain_continuous(examples, target_attribute,
                    attributes, attr, all_parameters):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """

    # find index of the chosen attr
    i = all_parameters.index(attr)
    value_list = np.sort(np.array(examples).transpose()[i])
    min_value = value_list[0]
    max_value = value_list[-1]
    range_value = max_value - min_value
    STEP_SIZE = 5  # in percent
    divide = min_value + STEP_SIZE * range_value / 100
    best_divide = min_value
    best_subset_entropy = 0.0

    while (divide < max_value - STEP_SIZE * range_value / 100):
        less_subset = []
        more_subset = []
        for example in examples:
            if (example[i] < divide):
                less_subset.append(example)
            else:
                more_subset.append(example)
        subset_entropy = 0.0
        subset_entropy += len(less_subset) * \
            entropy(less_subset, target_attribute, attributes,
                    all_parameters) / len(examples)
        subset_entropy += len(more_subset) * \
            entropy(more_subset, target_attribute, attributes,
                    all_parameters) / len(examples)

        if (best_subset_entropy < subset_entropy):
            best_subset_entropy = subset_entropy
            best_divide = divide

        divide += STEP_SIZE * range_value / 100

    return (entropy(examples, target_attribute, attributes, all_parameters) -
            best_subset_entropy), best_divide


def find_best_attribute(examples, target_attribute,
                        attributes, all_parameters):
    best_attribute = attributes[0]
    max_gain = 0
    best_divide = None
    for a in attributes:
        if a == target_attribute or a == "id":
            continue

        attr_index = all_parameters.index(a)
        options = np.unique(np.array(examples).transpose()
                            [attr_index]).shape[0]
        if options < 50:
            newGain = gain_discrete(
                examples, target_attribute, attributes, a, all_parameters)
            divide = None  # Only for continuos
        else:
            newGain, divide = gain_continuous(
                examples, target_attribute, attributes, a, all_parameters)
        if newGain > max_gain:
            max_gain = newGain
            best_attribute = a
            best_divide = divide
    return best_attribute, best_divide


def ID3(examples, target_attribute, attributes,
        depth, MAX_DEPTH, all_parameters):

    # Create root node
    root = Node()
    target_index = all_parameters.index(target_attribute)

    # Return single node, if all examples have same class
    # Find list of unique values in target_attribute of examples
    target_list = np.unique(np.array(examples).transpose()[target_index])

    if target_list.shape[0] == 1:
        root.answer = target_list[0]
        return root

    # If attributes is empty, return tree with most common value
    if len(attributes) == 0 or (depth >= MAX_DEPTH and MAX_DEPTH != -1):
        root.answer = np.argmax(np.bincount(
            np.array(examples).transpose()[target_index]))
        return root

    # Main algo
    best_attribute, divide = find_best_attribute(
        examples, target_attribute, attributes, all_parameters)

    root.attribute = best_attribute
    root.divide = divide

    if (root.divide is None):   # For discrete data
        attr_index = all_parameters.index(best_attribute)
        options = np.unique(np.array(examples).transpose()[attr_index])
        for value in options:
            example_subset = [example for example in
                              examples if example[attr_index] == value]
            branch = ID3(example_subset, target_attribute,
                         list(set(attributes) - set([best_attribute])),
                         depth + 1, MAX_DEPTH, all_parameters)
            branch.condition = value
            root.children.append(branch)
        # Add a default node, for values outside options
        # available in training data
        branch = Node()
        branch.answer = np.argmax(np.bincount(
            np.array(examples).transpose()[target_index]))
        root.children.append(branch)
    else:                       # For continuous data (divide not none)
        attr_index = all_parameters.index(best_attribute)
        less_subset = [
            example for example in examples if example[attr_index] < divide]
        branch = ID3(less_subset, target_attribute,
                     list(set(attributes) - set([best_attribute])),
                     depth + 1, MAX_DEPTH, all_parameters)
        branch.condition = 0
        root.children.append(branch)

        more_subset = [
            example for example in examples if example[attr_index] >= divide]
        branch = ID3(more_subset, target_attribute,
                     list(set(attributes) - set([best_attribute])),
                     depth + 1, MAX_DEPTH, all_parameters)
        branch.condition = 1
        root.children.append(branch)

    return root


def make_prediction(test_case, tree, attributes):
    if tree.attribute == "Leaf":
        return tree.answer

    attr_index = attributes.index(tree.attribute)
    val = test_case[attr_index]
    if (tree.divide is None):   # Discrete Values
        for child in tree.children:
            if (child.condition is None):
                none_child = child
            if (val == child.condition):
                return make_prediction(test_case, child, attributes)
        return make_prediction(test_case, none_child, attributes)

    else:                       # Continuous Values
        if (val < tree.divide):
            ans = 0
        else:
            ans = 1
        for child in tree.children:
            if (child.condition == ans):
                return make_prediction(test_case, child, attributes)


if __name__ == '__main__':
    train_file, test_file, MAX_DEPTH = checkArgs()

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

    tree = ID3(examples, "price",
               list(set(parameters) - set(['price', 'id'])),
               0, MAX_DEPTH, parameters)

    # Make predictions
    test_data, para = readCSV(test_file)
    price = []

    test_cases = [[0]] * len(para)

    for i in xrange(0, len(para)):
        # test_cases[i] = [float(x) for x in train_data[para[i]]]
        if (para[i] == "floors"):
            test_cases[i] = [int(2 * float(x)) for x in test_data[para[i]]]
        else:
            test_cases[i] = [int(x) for x in test_data[para[i]]]

    test_cases = np.array(test_cases).transpose()

    price = []
    for test_case in test_cases:
        price.append(make_prediction(test_case, tree, para))

    ids = [int(x) for x in test_data['id']]
    out = np.asarray([ids, price])
    np.savetxt("submission.csv", out.transpose(), '%d',
               delimiter=",", header="id,price", comments='')

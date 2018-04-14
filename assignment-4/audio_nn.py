import tensorflow as tf
import numpy as np
import sys
import csv

HIDDEN_LAYER_NODES = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 500
EPOCHS = 20


def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        # declare the training data placeholders
        # input x - 12 avg + 78 cov = 90
        x = tf.placeholder(tf.float32, [None, 90])
        # now declare the output data placeholder - 9 labels
        y = tf.placeholder(tf.float32, [None, 9])

        # now declare the weights connecting the input to the hidden layer
        W1 = tf.Variable(tf.random_normal(
            [90, HIDDEN_LAYER_NODES], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES]), name='b1')
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal(
            [HIDDEN_LAYER_NODES, 9], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([9]), name='b2')

        # calculate the output of the hidden layer
        hidden_out = tf.add(tf.matmul(x, W1), b1)
        hidden_out = tf.nn.relu(hidden_out)

        # output layer
        y_ = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))

        y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum
                                        (y * tf.log(y_clipped) +
                                         (1 - y) * tf.log(1 - y_clipped),
                                         axis=1))

        # add an optimiser
        optimiser = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).minimize(cross_entropy)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return graph, optimiser, init_op, accuracy, x, y, cross_entropy, y_


def train(train_examples, train_labels, test_examples, test_labels, test_samples):
    graph, optimiser, init_op, accuracy, x, y, cross_entropy, y_ = create_graph()
    session_conf = tf.ConfigProto(
        # device_count={
        #     "GPU":0
        #     }
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ),
    )
    with tf.Session(graph=graph, config=session_conf) as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(train_examples) / BATCH_SIZE)
        for epoch in range(EPOCHS):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = next_batch(
                    BATCH_SIZE, train_examples, train_labels)
                _, c = sess.run([optimiser, cross_entropy],
                                feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
                print("i: {}, avg_cost:{}".format(i, avg_cost))
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        print(sess.run(accuracy, feed_dict={
              x: test_examples, y: test_labels}))

        test_prediction = sess.run(y_, feed_dict={x: test_samples})
        print test_prediction
        # decoded_predictions = [np.where(r == 1)[0][0] for r in test_prediction]
        # out = np.asarray([ids, decoded_predictions])
        # np.savetxt("submission.csv", out.transpose(), '%d',
        #            delimiter=",", header="id,labels", comments='')


def checkArgs():
    if (len(sys.argv) != 4):
        print "Please enter three arguments. For instance, run: \
        \npython audio_nn.py train_class.csv dev_class.csv test-class.csv"
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


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = np.asarray([labels[i] for i in idx])

    return np.asarray(data_shuffle), one_hot(labels_shuffle)


def one_hot(indices, depth=9):
    one_hot_labels = np.zeros((len(indices), depth))
    one_hot_labels[np.arange(len(indices)), indices] = 1

    return one_hot_labels


if __name__ == "__main__":
    # Check args
    train_file, dev_file, test_file = checkArgs()

    # Read training set data
    train_data, parameters = readCSV(train_file)
    train_examples = [[0]] * (len(parameters))
    for i in xrange(0, len(parameters)):
        train_examples[i] = [float(x) for x in train_data[parameters[i]]]
    train_labels = np.array(train_examples[0]).astype(int)
    train_examples = np.array(train_examples[1:]).transpose()
    print("Completed reading training data")

    # Read dev set data
    dev_data, parameters = readCSV(dev_file)
    test_examples = [[0]] * (len(parameters))
    for i in xrange(0, len(parameters)):
        test_examples[i] = [float(x) for x in dev_data[parameters[i]]]
    test_labels = np.array(test_examples[0]).astype(int)
    test_examples = np.array(test_examples[1:]).transpose()
    print("Completed reading dev data")

    # Read dev set data
    test_data, parameters = readCSV(test_file)
    test_samples = [[0]] * (len(parameters))
    for i in xrange(0, len(parameters)):
        test_samples[i] = [float(x) for x in test_data[parameters[i]]]
    ids = np.array(test_samples[0]).astype(int)
    test_samples = np.array(test_samples[1:]).transpose()
    print("Completed reading test data")

    train(train_examples, train_labels, test_examples,
          one_hot(test_labels), test_samples)

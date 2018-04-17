import tensorflow as tf
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt


# Hyper parameters
HIDDEN_LAYER_NODES_OPTIONS = [250]
DROPOUT_RATE_OPTIONS = [0.99]
LEARNING_RATE = 0.00001
BATCH_SIZE = 500
EPOCHS = 20

# Changed internally, do not change
DROPOUT_RATE = 1.0
HIDDEN_LAYER_NODES = 100

SAVE_PATH = "./checkpoint-class/save/model.ckpt"
SAVE_FILE = "./predictions-class/submission.csv"

mins = [0] * 91
ptps = [0] * 91

DEGREE = 1
MULTIPLY_FEATURES = False
INPUT_NODES = 90


def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        # declare the training data placeholders
        # input x - 12 avg + 78 cov = 90
        x = tf.placeholder(tf.float32, [None, INPUT_NODES])
        # now declare the output data placeholdedr - 9 labels
        y = tf.placeholder(tf.float32, [None, 9])

        # Dropout keep probability
        keep_prob = tf.placeholder(tf.float32)

        W1 = tf.Variable(tf.random_normal(
            [INPUT_NODES, HIDDEN_LAYER_NODES], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES]), name='b1')

        W3 = tf.Variable(tf.random_normal(
            [HIDDEN_LAYER_NODES, 9], stddev=0.03), name='W3')
        b3 = tf.Variable(tf.random_normal([9]), name='b3')

        W2 = tf.Variable(tf.random_normal(
            [HIDDEN_LAYER_NODES, HIDDEN_LAYER_NODES], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES]), name='b2')
        # calculate the output of the hidden layer
        hidden_out1 = tf.add(tf.matmul(x, W1), b1)
        hidden_out1 = tf.nn.relu(hidden_out1)

        hidden_out1 = tf.nn.dropout(hidden_out1, keep_prob)

        # calculate the output of the hidden layer
        hidden_out2 = tf.add(tf.matmul(hidden_out1, W2), b2)
        hidden_out2 = tf.nn.relu(hidden_out2)

        hidden_out2 = tf.nn.dropout(hidden_out2, keep_prob)

        # output layer
        y_ = tf.add(tf.matmul(hidden_out2, W3), b3)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
        # pos_weight = tf.constant([1.0/86, 1.0/74, 1.0/123, 1.0/1140, 1.0/4412, 1.0/7738, 1.0/15028, 1.0/43602, 1.0/77797])))

        # y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        # cross_entropy = -tf.reduce_mean(tf.reduce_sum
        #                                 (y * tf.log(y_clipped) +
        #                                  (1 - y) * tf.log(1 - y_clipped),
        #                                  axis=1))

        # add an optimiser
        optimiser = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).minimize(cross_entropy)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return graph, optimiser, init_op, accuracy, x, y,  \
        cross_entropy, y_, keep_prob


def initialise_plot():
    plt.ion()
    plt.show()
    plt.gcf().clear()
    plt.title('HLN={} LR={} BS={} DR={}'.format(
        HIDDEN_LAYER_NODES, LEARNING_RATE, BATCH_SIZE, DROPOUT_RATE))
    plt.xlabel('Epoch')
    plt.ylabel('MSE')


def plot_graph(train_accuracy, test_accuracy):
    plt.gca().set_color_cycle(['red', 'green'])
    plt.axis([0, len(train_accuracy) +
              1, min(train_accuracy + test_accuracy), max(train_accuracy + test_accuracy)])
    plt.plot(np.arange(1, len(train_accuracy) + 1), np.array(train_accuracy))
    plt.plot(np.arange(1, len(test_accuracy) + 1), np.array(test_accuracy))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.draw()
    plt.pause(0.0001)


def train(train_examples, train_labels, test_examples,
          test_labels, test_samples, ids):

    results = np.zeros((len(HIDDEN_LAYER_NODES_OPTIONS) *
                        len(DROPOUT_RATE_OPTIONS), 3))
    cnt = 0
    for hln in HIDDEN_LAYER_NODES_OPTIONS:
        for d in DROPOUT_RATE_OPTIONS:
            global HIDDEN_LAYER_NODES, DROPOUT_RATE
            HIDDEN_LAYER_NODES = hln
            DROPOUT_RATE = d
            SAVE_PATH = "./checkpoint-class/save/model.ckpt".format(
                hln, int(d * 10))
            SAVE_FILE = "./predictions-class/submission_{}_{}.csv".format(
                hln, int(d * 10))

            graph, optimiser, init_op, accuracy, x, y, \
                cross_entropy, y_, keep_prob = create_graph()
            session_conf = tf.ConfigProto(
                # device_count={
                #     "GPU":0
                #     }
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                ),
            )
            with tf.Session(graph=graph, config=session_conf) as sess:
                saver = tf.train.Saver()
                try:
                    saver.restore(sess, SAVE_PATH)
                    print("Model restored.")
                except:
                    # initialise the variables
                    sess.run(init_op)
                    print("Model initialised.")
                total_batch = int(len(train_examples) / BATCH_SIZE)
                train_accuracy = []
                test_accuracy = []
                initialise_plot()
                for epoch in range(EPOCHS):
                    avg_cost = 0
                    for i in range(total_batch):
                        batch_x, batch_y = next_batch(
                            BATCH_SIZE, train_examples, train_labels)
                        _, c = sess.run([optimiser, cross_entropy],
                                        feed_dict={x: batch_x,
                                                   y: batch_y,
                                                   keep_prob: d})
                        avg_cost += c / total_batch
                        # print("i: {}, avg_cost:{}".format(i, avg_cost))

                    train_acc = sess.run(accuracy, feed_dict={
                        x: train_examples,
                        y: one_hot(train_labels),
                        keep_prob: 1.0})

                    test_acc = sess.run(accuracy, feed_dict={
                        x: test_examples,
                        y: one_hot(test_labels),
                        keep_prob: 1.0})

                    print("Epoch:{}\tTrain Acc={:.3f}\tTest Acc={:.3f}".format(
                        (epoch + 1), train_acc, test_acc))

                    train_accuracy.append(train_acc)
                    test_accuracy.append(test_acc)

                    plot_graph(train_accuracy, test_accuracy)

                    # print ans
                    if ((epoch + 1) % 100 == 0):
                        save_path = saver.save(sess, SAVE_PATH)
                        print("Model saved in path: %s" % save_path)

                print(test_acc)

                # store the data
                results[cnt, 0] = test_acc
                results[cnt, 1] = hln
                results[cnt, 2] = d
                cnt += 1

                save_path = saver.save(sess, SAVE_PATH)
                print("Model saved in path: %s" % save_path)

                plt.savefig('./images-class/{}_{}_{}_{}.png'.format(hln, d,
                                                                    LEARNING_RATE,
                                                                    BATCH_SIZE),
                            bbox_inches='tight')

                test_prediction = sess.run(
                    y_, feed_dict={x: test_samples, keep_prob: 1.0})
                decoded_predictions = np.argmax(test_prediction, axis=1)
                print decoded_predictions

                out = np.asarray([ids, decoded_predictions])
                np.savetxt(SAVE_FILE, out.transpose(), '%d',
                           delimiter=",", header="ids,label", comments='')

    for result in results:
        print("Accuracy: {:.3f}\t HIDDEN_LAYER_NODES: {}\t" +
              "DROPOUT: {}").format(result[0], result[1], result[2])


def checkArgs():
    if (len(sys.argv) != 4):
        print "Please enter three arguments. For instance, run: \
        \npython adv_audio_nn.py train_class.csv dev_class.csv test-class.csv"
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


def concat(X, Y):
    if X.size:
        return np.concatenate([X, Y])
    else:
        return Y


INDICES = []


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''

    # data_shuffle = np.array([])
    # labels_shuffle = np.array([])
    # sets = num / 9
    # remainder = num % 9

    # for i in range(0, 8):

    #     idx = np.arange(INDICES[i], INDICES[i + 1])
    #     values = np.random.choice(idx, sets, replace=True)
    #     to_concat = np.array([data[i] for i in values])
    #     data_shuffle = concat(data_shuffle, to_concat)
    #     to_concat = np.array([labels[i] for i in values])
    #     labels_shuffle = concat(labels_shuffle, to_concat)

    # idx = np.arange(INDICES[8], INDICES[9])
    # values = np.random.choice(idx, sets + remainder, replace=True)
    # to_concat = np.array([data[i] for i in values])
    # data_shuffle = concat(data_shuffle, to_concat)
    # to_concat = np.array([labels[i] for i in values])
    # labels_shuffle = concat(labels_shuffle, to_concat)

    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = np.asarray([labels[i] for i in idx])

    return data_shuffle, one_hot(labels_shuffle)


def one_hot(indices, depth=9):
    one_hot_labels = np.zeros((len(indices), depth))
    one_hot_labels[np.arange(len(indices)), indices] = 1

    return one_hot_labels


if __name__ == "__main__":
    # Check args
    train_file, dev_file, test_file = checkArgs()

    num_features = 90

    ########################
    # Read training set data
    train_data, parameters = readCSV(train_file)
    train_examples = [[0]] * (len(parameters))
    train_labels = np.array([int(x) for x in train_data[parameters[0]]])
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

    labels_idx_asc = train_labels.argsort()
    train_examples = train_examples[labels_idx_asc]
    train_labels = train_labels[labels_idx_asc]

    for i in xrange(0, 9):
        INDICES.append(np.argmax(train_labels == i))
    INDICES.append(len(train_labels))

    print("Completed reading training data")

    ###################
    # Read dev set data
    dev_data, parameters = readCSV(dev_file)
    test_examples = [[0]] * (len(parameters))
    test_labels = np.array([int(x) for x in dev_data[parameters[0]]])
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

    labels_idx_asc = test_labels.argsort()
    test_examples = test_examples[labels_idx_asc]
    test_labels = test_labels[labels_idx_asc]

    IND = []

    for i in xrange(0, 9):
        IND.append(np.argmax(test_labels == i))
    IND.append(len(test_labels))

    print IND
    # test_labels = test_labels[:5000]
    # test_examples = test_examples[:5000]
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

    train(train_examples, train_labels, test_examples,
          test_labels, test_samples, ids)

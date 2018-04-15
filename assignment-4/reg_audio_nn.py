import tensorflow as tf
import numpy as np
import sys
import csv

HIDDEN_LAYER_NODES = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 5000
EPOCHS = 1000
DROPOUT_RATE = 1.0


mins = [0] * 91
ptps = [0] * 91


def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        # declare the training data placeholders
        # input x - 12 avg + 78 cov = 90
        x = tf.placeholder(tf.float32, [None, 90])
        # now declare the output data placeholder - 9 labels
        y = tf.placeholder(tf.float32, [None, 1])

        # Dropout keep probability
        keep_prob = tf.placeholder(tf.float32)

        W1 = tf.Variable(tf.random_normal(
            [90, HIDDEN_LAYER_NODES], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES]), name='b1')

        W3 = tf.Variable(tf.random_normal(
            [HIDDEN_LAYER_NODES, 1], stddev=0.03), name='W3')
        b3 = tf.Variable(tf.random_normal([1]), name='b3')

        # W2 = tf.Variable(tf.random_normal(
        #     [HIDDEN_LAYER_NODES, HIDDEN_LAYER_NODES], stddev=0.03), name='W2')
        # b2 = tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES]), name='b2')
        # # calculate the output of the hidden layer
        # hidden_out1 = tf.add(tf.matmul(x, W1), b1)
        # hidden_out1 = tf.nn.relu(hidden_out1)

        # hidden_out1 = tf.nn.dropout(hidden_out1, keep_prob)

        # calculate the output of the hidden layer
        hidden_out2 = tf.add(tf.matmul(x, W1), b1)
        hidden_out2 = tf.nn.relu(hidden_out2)

        hidden_out2 = tf.nn.dropout(hidden_out2, keep_prob)

        # output layer
        y_ = tf.add(tf.matmul(hidden_out2, W3), b3)

        average_loss = tf.losses.mean_squared_error(y, y_)

        # y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        # cross_entropy = -tf.reduce_mean(tf.reduce_sum
        #                                 (y * tf.log(y_clipped) +
        #                                  (1 - y) * tf.log(1 - y_clipped),
        #                                  axis=1))

        # add an optimiser
        optimiser = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).minimize(average_loss)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # define an accuracy assessment operation
        accuracy = tf.losses.mean_squared_error(y, y_)

    return graph, optimiser, init_op, accuracy, x, y, average_loss, y_, keep_prob


def train(train_examples, train_labels, test_examples, test_labels, test_samples, ids):
    graph, optimiser, init_op, accuracy, x, y, average_loss, y_, keep_prob = create_graph()
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
            saver.restore(sess, "./save-reg/model.ckpt")
            print("Model restored.")
        except:
            # initialise the variables
            sess.run(init_op)
            print("Model initialised.")
        total_batch = int(len(train_examples) / BATCH_SIZE)
        for epoch in range(EPOCHS):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = next_batch(
                    BATCH_SIZE, train_examples, train_labels)
                ans, _, c = sess.run([y_, optimiser, average_loss],
                                     feed_dict={x: batch_x, y: batch_y, keep_prob: DROPOUT_RATE})
                avg_cost += c / total_batch
                # print("i: {}, avg_cost:{}".format(i, avg_cost))
            print("Epoch:{} cost={:.3f}".format((epoch + 1), avg_cost))
            # print ans

        save_path = saver.save(sess, "./save-reg/model.ckpt")
        print("Model saved in path: %s" % save_path)

        print(sess.run(accuracy, feed_dict={
              x: test_examples, y: np.reshape(test_labels, (-1, 1)), keep_prob: 1.0}))

        test_prediction = sess.run(
            y_, feed_dict={x: test_samples, keep_prob: 1.0})

        test_prediction = np.clip(np.reshape(test_prediction, -1), 1922, 2010)

        out = np.asarray([ids, test_prediction])
        np.savetxt("submission-reg.csv", out.transpose(), '%d',
                   delimiter=",", header="ids,label", comments='')


def checkArgs():
    if (len(sys.argv) != 4):
        print "Please enter three arguments. For instance, run: \
        \npython reg_audio_nn.py train_reg.csv dev_reg.csv test-reg.csv"
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

    return np.asarray(data_shuffle), np.reshape(labels_shuffle, (-1, 1))


if __name__ == "__main__":
    # Check args
    train_file, dev_file, test_file = checkArgs()

    # Read training set data
    train_data, parameters = readCSV(train_file)
    train_examples = [[0]] * (len(parameters))
    train_labels = np.array([float(x) for x in train_data[parameters[0]]])
    for i in xrange(1, len(parameters)):
        train_examples[i] = [float(x) for x in train_data[parameters[i]]]
        mins[i], ptps[i] = np.min(train_examples[i]), np.ptp(train_examples[i])
        train_examples[i] = [(float(x) - mins[i]) / ptps[i]
                             for x in train_data[parameters[i]]]
    train_examples = np.array(train_examples[1:]).transpose()
    print("Completed reading training data")

    # Read dev set data
    dev_data, parameters = readCSV(dev_file)
    test_examples = [[0]] * (len(parameters))
    test_labels = np.array([float(x) for x in dev_data[parameters[0]]])
    for i in xrange(1, len(parameters)):
        test_examples[i] = [(float(x) - mins[i]) / ptps[i]
                            for x in dev_data[parameters[i]]]
    test_examples = np.array(test_examples[1:]).transpose()
    print("Completed reading dev data")

    # Read dev set data
    test_data, parameters = readCSV(test_file)
    test_samples = [[0]] * (len(parameters))
    ids = np.array([float(x) for x in test_data[parameters[0]]]).astype(int)
    for i in xrange(1, len(parameters)):
        test_samples[i] = [(float(x) - mins[i]) / ptps[i]
                           for x in test_data[parameters[i]]]
    test_samples = np.array(test_samples[1:]).transpose()
    print("Completed reading test data")

    train(train_examples, train_labels, test_examples,
          test_labels, test_samples, ids)

import tensorflow as tf
import numpy as np
import sys
import csv
import logging
import matplotlib.pyplot as plt


TRAINING = True
WITHPLOT = False

mins = [0] * 91
ptps = [0] * 91


TimbreAvg1 = tf.feature_column.numeric_column("TimbreAvg1")
TimbreAvg2 = tf.feature_column.numeric_column("TimbreAvg2")
TimbreAvg3 = tf.feature_column.numeric_column("TimbreAvg3")
TimbreAvg4 = tf.feature_column.numeric_column("TimbreAvg4")
TimbreAvg5 = tf.feature_column.numeric_column("TimbreAvg5")
TimbreAvg6 = tf.feature_column.numeric_column("TimbreAvg6")
TimbreAvg7 = tf.feature_column.numeric_column("TimbreAvg7")
TimbreAvg8 = tf.feature_column.numeric_column("TimbreAvg8")
TimbreAvg9 = tf.feature_column.numeric_column("TimbreAvg9")
TimbreAvg10 = tf.feature_column.numeric_column("TimbreAvg10")
TimbreAvg11 = tf.feature_column.numeric_column("TimbreAvg11")
TimbreAvg12 = tf.feature_column.numeric_column("TimbreAvg12")

TimbreCovariance1 = tf.feature_column.numeric_column("TimbreCovariance1")
TimbreCovariance2 = tf.feature_column.numeric_column("TimbreCovariance2")
TimbreCovariance3 = tf.feature_column.numeric_column("TimbreCovariance3")
TimbreCovariance4 = tf.feature_column.numeric_column("TimbreCovariance4")
TimbreCovariance5 = tf.feature_column.numeric_column("TimbreCovariance5")
TimbreCovariance6 = tf.feature_column.numeric_column("TimbreCovariance6")
TimbreCovariance7 = tf.feature_column.numeric_column("TimbreCovariance7")
TimbreCovariance8 = tf.feature_column.numeric_column("TimbreCovariance8")
TimbreCovariance9 = tf.feature_column.numeric_column("TimbreCovariance9")

TimbreCovariance10 = tf.feature_column.numeric_column("TimbreCovariance10")
TimbreCovariance11 = tf.feature_column.numeric_column("TimbreCovariance11")
TimbreCovariance12 = tf.feature_column.numeric_column("TimbreCovariance12")
TimbreCovariance13 = tf.feature_column.numeric_column("TimbreCovariance13")
TimbreCovariance14 = tf.feature_column.numeric_column("TimbreCovariance14")
TimbreCovariance15 = tf.feature_column.numeric_column("TimbreCovariance15")
TimbreCovariance16 = tf.feature_column.numeric_column("TimbreCovariance16")
TimbreCovariance17 = tf.feature_column.numeric_column("TimbreCovariance17")
TimbreCovariance18 = tf.feature_column.numeric_column("TimbreCovariance18")
TimbreCovariance19 = tf.feature_column.numeric_column("TimbreCovariance19")

TimbreCovariance20 = tf.feature_column.numeric_column("TimbreCovariance20")
TimbreCovariance21 = tf.feature_column.numeric_column("TimbreCovariance21")
TimbreCovariance22 = tf.feature_column.numeric_column("TimbreCovariance22")
TimbreCovariance23 = tf.feature_column.numeric_column("TimbreCovariance23")
TimbreCovariance24 = tf.feature_column.numeric_column("TimbreCovariance24")
TimbreCovariance25 = tf.feature_column.numeric_column("TimbreCovariance25")
TimbreCovariance26 = tf.feature_column.numeric_column("TimbreCovariance26")
TimbreCovariance27 = tf.feature_column.numeric_column("TimbreCovariance27")
TimbreCovariance28 = tf.feature_column.numeric_column("TimbreCovariance28")
TimbreCovariance29 = tf.feature_column.numeric_column("TimbreCovariance29")

TimbreCovariance30 = tf.feature_column.numeric_column("TimbreCovariance30")
TimbreCovariance31 = tf.feature_column.numeric_column("TimbreCovariance31")
TimbreCovariance32 = tf.feature_column.numeric_column("TimbreCovariance32")
TimbreCovariance33 = tf.feature_column.numeric_column("TimbreCovariance33")
TimbreCovariance34 = tf.feature_column.numeric_column("TimbreCovariance34")
TimbreCovariance35 = tf.feature_column.numeric_column("TimbreCovariance35")
TimbreCovariance36 = tf.feature_column.numeric_column("TimbreCovariance36")
TimbreCovariance37 = tf.feature_column.numeric_column("TimbreCovariance37")
TimbreCovariance38 = tf.feature_column.numeric_column("TimbreCovariance38")
TimbreCovariance39 = tf.feature_column.numeric_column("TimbreCovariance39")

TimbreCovariance40 = tf.feature_column.numeric_column("TimbreCovariance40")
TimbreCovariance41 = tf.feature_column.numeric_column("TimbreCovariance41")
TimbreCovariance42 = tf.feature_column.numeric_column("TimbreCovariance42")
TimbreCovariance43 = tf.feature_column.numeric_column("TimbreCovariance43")
TimbreCovariance44 = tf.feature_column.numeric_column("TimbreCovariance44")
TimbreCovariance45 = tf.feature_column.numeric_column("TimbreCovariance45")
TimbreCovariance46 = tf.feature_column.numeric_column("TimbreCovariance46")
TimbreCovariance47 = tf.feature_column.numeric_column("TimbreCovariance47")
TimbreCovariance48 = tf.feature_column.numeric_column("TimbreCovariance48")
TimbreCovariance49 = tf.feature_column.numeric_column("TimbreCovariance49")

TimbreCovariance50 = tf.feature_column.numeric_column("TimbreCovariance50")
TimbreCovariance51 = tf.feature_column.numeric_column("TimbreCovariance51")
TimbreCovariance52 = tf.feature_column.numeric_column("TimbreCovariance52")
TimbreCovariance53 = tf.feature_column.numeric_column("TimbreCovariance53")
TimbreCovariance54 = tf.feature_column.numeric_column("TimbreCovariance54")
TimbreCovariance55 = tf.feature_column.numeric_column("TimbreCovariance55")
TimbreCovariance56 = tf.feature_column.numeric_column("TimbreCovariance56")
TimbreCovariance57 = tf.feature_column.numeric_column("TimbreCovariance57")
TimbreCovariance58 = tf.feature_column.numeric_column("TimbreCovariance58")
TimbreCovariance59 = tf.feature_column.numeric_column("TimbreCovariance59")

TimbreCovariance60 = tf.feature_column.numeric_column("TimbreCovariance60")
TimbreCovariance61 = tf.feature_column.numeric_column("TimbreCovariance61")
TimbreCovariance62 = tf.feature_column.numeric_column("TimbreCovariance62")
TimbreCovariance63 = tf.feature_column.numeric_column("TimbreCovariance63")
TimbreCovariance64 = tf.feature_column.numeric_column("TimbreCovariance64")
TimbreCovariance65 = tf.feature_column.numeric_column("TimbreCovariance65")
TimbreCovariance66 = tf.feature_column.numeric_column("TimbreCovariance66")
TimbreCovariance67 = tf.feature_column.numeric_column("TimbreCovariance67")
TimbreCovariance68 = tf.feature_column.numeric_column("TimbreCovariance68")
TimbreCovariance69 = tf.feature_column.numeric_column("TimbreCovariance69")

TimbreCovariance70 = tf.feature_column.numeric_column("TimbreCovariance70")
TimbreCovariance71 = tf.feature_column.numeric_column("TimbreCovariance71")
TimbreCovariance72 = tf.feature_column.numeric_column("TimbreCovariance72")
TimbreCovariance73 = tf.feature_column.numeric_column("TimbreCovariance73")
TimbreCovariance74 = tf.feature_column.numeric_column("TimbreCovariance74")
TimbreCovariance75 = tf.feature_column.numeric_column("TimbreCovariance75")
TimbreCovariance76 = tf.feature_column.numeric_column("TimbreCovariance76")
TimbreCovariance77 = tf.feature_column.numeric_column("TimbreCovariance77")
TimbreCovariance78 = tf.feature_column.numeric_column("TimbreCovariance78")

feature_columns = {TimbreAvg1, TimbreAvg2, TimbreAvg3, TimbreAvg4, TimbreAvg5,
                   TimbreAvg6, TimbreAvg7, TimbreAvg8, TimbreAvg9, TimbreAvg10,
                   TimbreAvg11, TimbreAvg12,
                   TimbreCovariance1, TimbreCovariance2, TimbreCovariance3,
                   TimbreCovariance4, TimbreCovariance5, TimbreCovariance6,
                   TimbreCovariance7, TimbreCovariance8, TimbreCovariance9,
                   TimbreCovariance10, TimbreCovariance11, TimbreCovariance12,
                   TimbreCovariance13, TimbreCovariance14, TimbreCovariance15,
                   TimbreCovariance16, TimbreCovariance17, TimbreCovariance18,
                   TimbreCovariance19, TimbreCovariance20, TimbreCovariance21,
                   TimbreCovariance22, TimbreCovariance23, TimbreCovariance24,
                   TimbreCovariance25, TimbreCovariance26, TimbreCovariance27,
                   TimbreCovariance28, TimbreCovariance29, TimbreCovariance30,
                   TimbreCovariance31, TimbreCovariance32, TimbreCovariance33,
                   TimbreCovariance34, TimbreCovariance35, TimbreCovariance36,
                   TimbreCovariance37, TimbreCovariance38, TimbreCovariance39,
                   TimbreCovariance40, TimbreCovariance41, TimbreCovariance42,
                   TimbreCovariance43, TimbreCovariance44, TimbreCovariance45,
                   TimbreCovariance46, TimbreCovariance47, TimbreCovariance48,
                   TimbreCovariance49, TimbreCovariance50, TimbreCovariance51,
                   TimbreCovariance52, TimbreCovariance53, TimbreCovariance54,
                   TimbreCovariance55, TimbreCovariance56, TimbreCovariance57,
                   TimbreCovariance58, TimbreCovariance59, TimbreCovariance60,
                   TimbreCovariance61, TimbreCovariance62, TimbreCovariance63,
                   TimbreCovariance64, TimbreCovariance65, TimbreCovariance66,
                   TimbreCovariance67, TimbreCovariance68, TimbreCovariance69,
                   TimbreCovariance70, TimbreCovariance71, TimbreCovariance72,
                   TimbreCovariance73, TimbreCovariance74, TimbreCovariance75,
                   TimbreCovariance76, TimbreCovariance77, TimbreCovariance78}


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


# Defining the Tensorflow input functions
# for training
def training_input_fn(batch_size=1):
    return tf.estimator.inputs.numpy_input_fn(
        x=train_dict,
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)


# for test
def test_input_fn():
    return tf.estimator.inputs.numpy_input_fn(
        x=dev_dict,
        y=test_labels,
        num_epochs=1,
        shuffle=False)


# for prediction
def prediction_input_fn():
    return tf.estimator.inputs.numpy_input_fn(
        x=test_dict,
        y=None,
        num_epochs=1,
        shuffle=False)


# Check args
train_file, dev_file, test_file = checkArgs()

# Read training set data
train_dict = {}
train_data, parameters = readCSV(train_file)
train_examples = [[0]] * (len(parameters))
train_labels = np.array([float(x) for x in train_data[parameters[0]]])
for i in xrange(1, len(parameters)):
    train_examples[i] = [float(x) for x in train_data[parameters[i]]]
    mins[i], ptps[i] = np.min(train_examples[i]), np.ptp(train_examples[i])
    train_examples[i] = np.array([(float(x) - mins[i]) / ptps[i]
                                  for x in train_data[parameters[i]]])
    train_dict[parameters[i]] = train_examples[i]
train_examples = np.array(train_examples[1:]).transpose()
print("Completed reading training data")

# Read dev set data
dev_dict = {}
dev_data, parameters = readCSV(dev_file)
test_examples = [[0]] * (len(parameters))
test_labels = np.array([float(x) for x in dev_data[parameters[0]]])
for i in xrange(1, len(parameters)):
    test_examples[i] = np.array([(float(x) - mins[i]) / ptps[i]
                                 for x in dev_data[parameters[i]]])
    dev_dict[parameters[i]] = test_examples[i]
test_examples = np.array(test_examples[1:]).transpose()
print("Completed reading dev data")

# Read dev set data
test_data, parameters = readCSV(test_file)
test_dict = {}
test_samples = [[0]] * (len(parameters))
ids = np.array([float(x) for x in test_data[parameters[0]]]).astype(int)
for i in xrange(1, len(parameters)):
    test_samples[i] = np.array([(float(x) - mins[i]) / ptps[i]
                                for x in test_data[parameters[i]]])
    test_dict[parameters[i]] = test_samples[i]
test_samples = np.array(test_samples[1:]).transpose()
print("Completed reading test data")


STEPS_PER_EPOCH = 100
EPOCHS = 1
BATCH_SIZE = 100

hidden_layers = [100]
dropout = 0.0

MODEL_PATH = './DNNRegressors/'
for hl in hidden_layers:
    MODEL_PATH += '%s_' % hl
MODEL_PATH += 'D0%s' % (int(dropout * 10))
logging.info('Saving to %s' % MODEL_PATH)


# Validation and Test ConfigurationS
test_config = tf.estimator.RunConfig(save_checkpoints_steps=100,
                                     save_checkpoints_secs=None)


regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                      label_dimension=1,
                                      hidden_units=hidden_layers,
                                      model_dir=MODEL_PATH,
                                      dropout=dropout,
                                      config=test_config)
for epoch in range(EPOCHS):
    regressor.train(input_fn=training_input_fn(batch_size=BATCH_SIZE),
                    steps=STEPS_PER_EPOCH)
    eval_dict = regressor.evaluate(input_fn=test_input_fn())
    print('%.5f Loss' % (eval_dict['loss']))

y_pred = regressor.predict(input_fn=prediction_input_fn())
predictions = np.clip(np.array(p["predictions"][0]
                               for p in y_pred), 1922, 2011)

out = np.asarray([ids, predictions])
np.savetxt("submission-reg_dnn.csv", out.transpose(), '%d',
           delimiter=",", header="ids,label", comments='')

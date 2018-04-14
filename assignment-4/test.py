import numpy as np


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0, len(data))
    print idx
    np.random.shuffle(idx)
    print idx
    idx = idx[:num]
    print idx
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def one_hot(indices, depth=9):
    one_hot_labels = np.zeros((len(indices), depth))
    one_hot_labels[np.arange(len(indices)), indices] = 1

    return one_hot_labels


a = np.asanyarray([[0, 1, 0, 5, 6, 8, 0, 0, 0], [440, 1, 0, 0, 0, 0, 0, 0, 0], [
                  0, 1, 0, 0, 120, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 99, 0, 0]])

b = np.argmax(a,axis=1)
print b+1

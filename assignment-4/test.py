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


Xtr, Ytr = np.arange(0, 10), np.arange(0, 100).reshape(10, 10)
print(Xtr)
print(Ytr)

Xtr, Ytr = next_batch(5, Xtr, Ytr)
print('\n5 random samples')
print(Xtr)
print(Ytr)

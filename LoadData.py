import numpy as np


class Load:
    def __init__(self, binary=False, bias=False, one_hot=False, normalize=True):
        self.binary = binary
        self.bias = bias
        self.one_hot = one_hot
        self.normalize = normalize

    def load_mnist(self):
        X = np.load('AvailableData/train_img.npy')
        Y = np.load('AvailableData/train_label.npy').astype(np.int)
        test_X = np.load('AvailableData/test_img.npy')
        test_Y = np.load('AvailableData/test_label.npy').astype(np.int)

        X = X.reshape(60000, 784)
        test_X = test_X.reshape(10000, 784)

        if self.normalize:
            x_min = np.min(X)
            x_max = np.max(X)
            test_x_min = np.min(test_X)
            test_x_max = np.max(test_X)

            X = (X - x_min) / (x_max - x_min)
            test_X = (test_X - test_x_min) / (test_x_max - test_x_min)

        if self.one_hot:
            tmp = np.zeros((Y.shape[0], 10))
            idx = np.arange(Y.shape[0])
            tmp[idx, Y[idx]] = 1
            Y = tmp

            tmp = np.zeros((test_Y.shape[0], 10))
            idx = np.arange(test_Y.shape[0])
            tmp[idx, test_Y[idx]] = 1
            test_Y = tmp

        if self.binary:
            ind = np.where(Y < 2)
            Y = Y[ind]
            X = X[ind]

            ind = np.where(test_Y < 2)
            test_Y = test_Y[ind]
            test_X = test_X[ind]

        if self.bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
            test_X = np.column_stack((np.ones(test_X.shape[0]), test_X))

        return X, Y, test_X, test_Y

    def load_iris(self, p=0.8):
        tmp = np.load('AvailableData/Iris_data.npy')

        if self.bias:
            tmp = np.column_stack((np.ones(tmp.shape[0]), tmp))

        np.random.shuffle(tmp)  # 打乱数据
        train = tmp[:int(p * tmp.shape[0]), :]
        test = tmp[int(p * tmp.shape[0]):, :]

        X = train[:, :-1]
        Y = train[:, -1].astype(np.int)
        test_X = test[:, :-1]
        test_Y = test[:, -1].astype(np.int)

        if self.normalize:
            x_min = np.min(X)
            x_max = np.max(X)
            test_x_min = np.min(test_X)
            test_x_max = np.max(test_X)

            X = (X - x_min) / (x_max - x_min)
            test_X = (test_X - test_x_min) / (test_x_max - test_x_min)

        if self.one_hot:
            tmp = np.zeros((Y.shape[0], 3))
            idx = np.arange(Y.shape[0])
            tmp[idx, Y[idx] - 1] = 1
            Y = tmp

            tmp = np.zeros((test_Y.shape[0], 3))
            idx = np.arange(test_Y.shape[0])
            tmp[idx, test_Y[idx] - 1] = 1
            test_Y = tmp

        return X, Y, test_X, test_Y

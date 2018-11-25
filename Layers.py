import numpy as np


class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.w.T)
        return dx  # return dx


class Relu:
    def __init__(self):
        self.flag = None

    def forward(self, x):
        self.flag = x <= 0
        return np.maximum(x, 0)

    def backward(self, dout):
        dout[self.flag] = 0
        return dout  # return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out) * self.out  # return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # one-hot
        self.t = None  # one-hot

    @staticmethod
    def softmax(x):
        max_xi = np.max(x, axis=1)
        # print(max_xi)
        exp_x = np.exp(x.T - max_xi)
        sum_exp_xi = np.sum(exp_x, axis=0)
        p = (exp_x / sum_exp_xi).T

        # exp_x = np.exp(x)
        # sum_exp_xi = np.sum(exp_x, axis=1)
        #
        # p = (exp_x.T - sum_exp_xi).T

        # print(p)
        return p

    @staticmethod
    def cross_entropy(y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta)) / t.shape[0]

    def forward(self, x, t):
        self.t = t  # one-hot
        self.y = self.softmax(x)
        self.loss = self.cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        dx = dout * (self.y - self.t) / self.t.shape[0]
        return dx


class Convolution:
    def __init__(self, w, b, stride=1, pad=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

    @staticmethod
    def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
        """
        Parameters
        ----------
        input_data : 由(数据量, 高, 长)的3维数组构成的输入数据(灰度图)
        filter_h : 滤波器的高
        filter_w : 滤波器的长
        stride : 步幅
        pad : 填充

        Returns
        -------
        col : 2维数组
        """
        N, H, W = input_data.shape
        out_h = int(1 + (H + 2 * pad - filter_h) / stride)
        out_w = int(1 + (W + 2 * pad - filter_w) / stride)

        img = np.pad(input_data, [(0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, y, x, :, :] = img[:, y:y_max:stride, x:x_max:stride]

        col = col.transpose((0, 3, 4, 1, 2)).reshape((N * out_h * out_w, -1))
        return col

    def forward(self, x):
        FN, FH, FW = self.w.shape
        N, H, W = x.shape

        out_h = int((1 + (H + 2 * self.pad - FH) / self.stride))
        out_w = int((1 + (W + 2 * self.pad - FW) / self.stride))

        col_x = self.im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.w.reshape((FN, -1)).T

        out = np.dot(col_x, col_w) + self.b
        out = out.reshape((N, out_h, out_w, -1)).transpose((0, 3, 1, 2))

        return out

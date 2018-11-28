from LoadData import *
from Layers import *
import matplotlib.pyplot as plt


class Network:
    def __init__(self, sizes):
        self.X, self.Y, self.test_X, self.test_Y = Load(one_hot=True).load_mnist()
        self.T = np.argmax(self.Y, axis=1)
        self.test_T = np.argmax(self.test_Y, axis=1)

        self.layers_num = len(sizes) - 1
        weight_init_std = 0.001

        # self.w = [np.ones((x, y)) for x, y in zip(sizes[:-1], sizes[1:])]
        # self.b = [np.ones((1, y)) for y in sizes[1:]]

        self.w = [weight_init_std * np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, y)) for y in sizes[1:]]

        self.layers = list()
        for i in range(self.layers_num - 1):
            self.layers.append(Affine(self.w[i], self.b[i]))
            self.layers.append(Relu())
        self.layers.append(Affine(self.w[-1], self.b[-1]))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # m*10 Matrix

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / t.shape[0]

    def gradient(self, x, t):
        l = self.loss(x, t)

        dout = self.last_layer.backward(1)

        layers = self.layers.copy()
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grad_w = [self.layers[2 * i].dw for i in range(self.layers_num)]
        grad_b = [self.layers[2 * i].db for i in range(self.layers_num)]

        return grad_w, grad_b

    def train(self):
        batch_size = 600
        train_size = self.X.shape[0]
        epoch_num = 10000
        lr = 0.02  # learning rate
        beta1 = 0.9
        beta2 = 0.999
        momentum = 1

        # v_w = [np.zeros_like(w) for w in self.w]
        # v_b = [np.zeros_like(b) for b in self.b]

        h_w = [np.zeros_like(w) for w in self.w]
        h_b = [np.zeros_like(b) for b in self.b]

        # m_w = v_w.copy()
        # m_b = v_b.copy()

        flag = max(int(train_size / batch_size), 1)

        train_loss_list = list()
        train_acc_list = list()
        test_loss_list = list()
        test_acc_list = list()
        plot_x = list()

        for i in range(epoch_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = self.X[batch_mask]
            y_batch = self.Y[batch_mask]

            grad_w, grad_b = self.gradient(x_batch, y_batch)

            # tmp_lr = lr * np.sqrt(1 - beta2 ** (i + 1)) / (1 - beta1 ** (i + 1))

            for j in range(self.layers_num):
                # Adam
                # m_w[j] += (1 - beta1) * (grad_w[j] - m_w[j])
                # m_b[j] += (1 - beta1) * (grad_b[j] - m_b[j])
                #
                # v_w[j] += (1 - beta2) * (grad_w[j] ** 2 - v_w[j])
                # v_b[j] += (1 - beta2) * (grad_b[j] ** 2 - v_b[j])
                #
                # self.layers[2 * j].w -= tmp_lr * m_w[j] / (np.sqrt(v_w[j] - np.min(v_w[j])) + 1e-7)
                # self.layers[2 * j].b -= tmp_lr * m_b[j] / (np.sqrt(v_b[j] - np.min(v_b[j])) + 1e-7)

                # Momentum
                # v_w[j] = momentum * v_w[j] - lr * grad_w[j]
                # v_b[j] = momentum * v_b[j] - lr * grad_b[j]
                #
                # self.layers[2 * j].w += v_w[j]
                # self.layers[2 * j].b += v_b[j]

                # AdaGrad
                # lr=0.1
                h_w[j] += np.square(grad_w[j])
                h_b[j] += np.square(grad_b[j])

                self.layers[2 * j].w -= lr * grad_w[j] / (np.sqrt(h_w[j]) + 1e-7)
                self.layers[2 * j].b -= lr * grad_b[j] / (np.sqrt(h_b[j]) + 1e-7)

            # test_acc = self.accuracy(self.test_X, self.test_Y)
            # # train_acc_list.append(test_acc)
            # print("第", i, "轮，loss-->", self.loss(x_batch, y_batch), "acc-->", test_acc)

            if i % flag == 0:
                plot_x.append(i)

                tmp = self.predict(self.X)
                train_loss_list.append(self.last_layer.forward(tmp, self.Y))
                tmp = np.argmax(tmp, axis=1)
                train_acc_list.append(np.sum(tmp == self.T) / self.T.shape[0])

                tmp = self.predict(self.test_X)
                test_loss = self.last_layer.forward(tmp, self.test_Y)
                tmp = np.argmax(tmp, axis=1)
                test_acc = np.sum(tmp == self.test_T) / self.test_T.shape[0]
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)

                print("第", i, "轮，test_loss-->", test_loss, "test_acc-->", test_acc)

        plot_x.append(epoch_num)

        tmp = self.predict(self.X)
        train_loss_list.append(self.last_layer.forward(tmp, self.Y))
        tmp = np.argmax(tmp, axis=1)
        train_acc_list.append(np.sum(tmp == self.T) / self.T.shape[0])

        tmp = self.predict(self.test_X)
        test_loss = self.last_layer.forward(tmp, self.test_Y)
        tmp = np.argmax(tmp, axis=1)
        test_acc = np.sum(tmp == self.test_T) / self.test_T.shape[0]
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print("第", epoch_num, "轮，test_loss-->", test_loss, "test_acc-->", test_acc)

        plt.plot(plot_x, train_loss_list, '--', label='train_loss')
        plt.plot(plot_x, test_loss_list, '--', label='test_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

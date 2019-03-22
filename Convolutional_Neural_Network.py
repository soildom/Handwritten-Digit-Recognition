from LoadData import *
from Layers import *
import matplotlib.pyplot as plt


class ConvNet:
    def __init__(self, sizes):
        filter_num = 30
        filter_size = 5
        filter_pad = 0
        filter_stride = 1

        pool_size = 2
        pool_stride = 2

        self.conv_input_size = 28
        self.conv_output_size = int(1 + (self.conv_input_size + 2 * filter_pad - filter_size) / filter_stride)
        self.pool_output_size = filter_num * (int(self.conv_output_size / pool_size) ** 2)

        self.X, self.Y, self.test_X, self.test_Y = Load(one_hot=True, flatten=True).load_mnist()
        self.T = np.argmax(self.Y, axis=1)
        self.test_T = np.argmax(self.test_Y, axis=1)
        sizes.insert(0, self.pool_output_size)

        weight_init_std = 0.01

        # print(sizes)

        self.conv_w = weight_init_std * np.random.randn(filter_num, filter_size, filter_size)
        self.conv_b = np.zeros((filter_num))

        self.w = [weight_init_std * np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, y)) for y in sizes[1:]]

        self.conv_layers = list()
        self.conv_layers.append(Convolution(self.conv_w, self.conv_b, filter_stride, filter_pad))
        self.conv_layers.append(Relu())
        self.conv_layers.append(Pooling(pool_size, pool_size, pool_stride))

        self.layers = list()
        self.affine_layers_num = len(sizes) - 1
        for i in range(self.affine_layers_num - 1):
            self.layers.append(Affine(self.w[i], self.b[i]))
            self.layers.append(Relu())
        self.layers.append(Affine(self.w[-1], self.b[-1]))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.conv_layers:
            # print(x.shape)
            x = layer.forward(x)
        x = x.reshape((x.shape[0], -1))
        for layer in self.layers:
            x = layer.forward(x)
        return x

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

        conv_layers = self.conv_layers.copy()
        conv_layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        for layer in conv_layers:
            dout = layer.backward(dout)

        grad_w = [self.layers[2 * i].dw for i in range(self.affine_layers_num)]
        grad_b = [self.layers[2 * i].db for i in range(self.affine_layers_num)]

        grad_conv_w = self.conv_layers[0].dw
        grad_conv_b = self.conv_layers[0].db

        return grad_w, grad_b, grad_conv_w, grad_conv_b

    def train(self):
        batch_size = 100
        train_size = self.X.shape[0]
        iteration_num = 10000
        lr = 0.01  # learning rate
        momentum = 1

        # v_w = [np.zeros_like(w) for w in self.w]
        # v_b = [np.zeros_like(b) for b in self.b]

        h_w = [np.zeros_like(w) for w in self.w]
        h_b = [np.zeros_like(b) for b in self.b]

        h_cw = np.zeros_like(self.conv_layers[0].w)
        h_cb = np.zeros_like(self.conv_layers[0].b)

        epoch = max(int(train_size / batch_size), 1)

        # train_loss_list = list()
        # train_acc_list = list()
        test_loss_list = list()
        test_acc_list = list()
        plot_x = list()

        for i in range(iteration_num):
            # print(i)
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = self.X[batch_mask]
            y_batch = self.Y[batch_mask]

            # print("计算梯度")

            grad_w, grad_b, grad_cw, grad_cb = self.gradient(x_batch, y_batch)

            # print("迭代")

            for j in range(self.affine_layers_num):
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

                # self.layers[2 * j].w -= lr * grad_w[j]
                # self.layers[2 * j].b -= lr * grad_b[j]

            h_cw += np.square(grad_cw)
            h_cb += np.square(grad_cb)

            self.conv_layers[0].w -= lr * grad_cw / (np.sqrt(h_cw) + 1e-7)
            self.conv_layers[0].b -= lr * grad_cb / (np.sqrt(h_cb) + 1e-7)

            # self.conv_layers[0].w -= lr * grad_cw
            # self.conv_layers[0].b -= lr * grad_cb

            if i % epoch == 0:
                plot_x.append(i)

                # tmp = self.predict(self.X)
                # train_loss_list.append(self.last_layer.forward(tmp, self.Y))
                # tmp = np.argmax(tmp, axis=1)
                # train_acc_list.append(np.sum(tmp == self.T) / self.T.shape[0])

                tmp = self.predict(self.test_X)
                test_loss = self.last_layer.forward(tmp, self.test_Y)
                tmp = np.argmax(tmp, axis=1)
                test_acc = np.sum(tmp == self.test_T) / self.test_T.shape[0]
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)

                print("第", i, "次迭代，test_loss-->", test_loss, "test_acc-->", test_acc)

        plot_x.append(iteration_num)

        # tmp = self.predict(self.X)
        # train_loss_list.append(self.last_layer.forward(tmp, self.Y))
        # tmp = np.argmax(tmp, axis=1)
        # train_acc_list.append(np.sum(tmp == self.T) / self.T.shape[0])

        tmp = self.predict(self.test_X)
        test_loss = self.last_layer.forward(tmp, self.test_Y)
        tmp = np.argmax(tmp, axis=1)
        test_acc = np.sum(tmp == self.test_T) / self.test_T.shape[0]
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print("第", iteration_num, "次迭代，test_loss-->", test_loss, "test_acc-->", test_acc)

        # plt.plot(plot_x, train_loss_list, '--', label='train_loss')
        plt.plot(plot_x, test_loss_list, '--', label='test_loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


net = ConvNet([100, 10])
# print(net.predict(net.test_X))
net.train()

##测试卷积
# x = np.arange(16)
# # np.random.shuffle(x)
# x = x.reshape((1, 4, 4))
# print(x)
# w = np.ones((2, 2, 2))
# b = np.ones((2))
# c = Convolution(w, b)
# y = c.forward(x)
# print()
# print(y)
# print()
# print(c.backward(y))
# print(c.dw)

##测试池化
# x = np.arange(96)
# np.random.shuffle(x)
# x = x.reshape((2, 3, 4, 4))
# p = Pooling(2, 2, 2)
# print(x)
# print()
# y=p.forward(x)
# print(y)
# print()
# print(p.backward(y))

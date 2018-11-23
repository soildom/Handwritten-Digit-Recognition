import numpy as np
import struct


class Mnist2npy:
    def __init__(self):
        label_name = ['RawData/t10k-labels-idx1-ubyte',
                      'RawData/train-labels-idx1-ubyte']
        img_name = ['RawData/t10k-images-idx3-ubyte',
                    'RawData/train-images-idx3-ubyte']

        train_img = self.get_img(img_name[1])
        train_label = self.get_label(label_name[1])
        test_img = self.get_img(img_name[0])
        test_label = self.get_label(label_name[0])

        np.save('AvailableData/train_img', train_img)
        np.save('AvailableData/train_label', train_label)
        np.save('AvailableData/test_img', test_img)
        np.save('AvailableData/test_label', test_label)

    def get_img(self, url):
        with open(url, 'rb') as f:
            buf = f.read()

        offset = 0
        magic, imageNum, rows, cols = struct.unpack_from('>IIII', buf, offset)
        offset += struct.calcsize('>IIII')
        images = np.empty((imageNum, rows, cols))
        image_size = rows * cols
        fmt = '>' + str(image_size) + 'B'

        for i in range(imageNum):
            images[i] = np.array(struct.unpack_from(fmt, buf, offset)).reshape((rows, cols))
            offset += struct.calcsize(fmt)

        return images

    def get_label(self, url):
        with open(url, 'rb') as f:
            buf = f.read()

        offset = 0
        magic, LabelNum = struct.unpack_from('>II', buf, offset)
        offset += struct.calcsize('>II')
        Labels = np.zeros((LabelNum))

        for i in range(LabelNum):
            Labels[i] = np.array(struct.unpack_from('>B', buf, offset))
            offset += struct.calcsize('>B')

        return Labels


Mnist2npy()

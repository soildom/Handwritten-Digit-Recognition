import time
from BP_Network_MultipleClassification import *

t0 = time.clock()

net = Network([784, 300, 100, 10])

net.train()
# print(net.accuracy(net.test_X, net.test_Y))

t = time.clock() - t0
print("\n耗时：", t, "s=", t / 60, "min")

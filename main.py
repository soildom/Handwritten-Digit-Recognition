import time
from FC_Neural_Network import *

t0 = time.clock()

net = Network([784, 158, 30, 10])

net.train()

t = time.clock() - t0
print("\n耗时：", t, "s=", t / 60, "min")

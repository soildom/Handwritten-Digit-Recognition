import time
from BP_Network_MultipleClassification import *

t0 = time.clock()

net = Network([784, 200, 150, 30, 10])

net.train()

t = time.clock() - t0
print("\n耗时：", t, "s=", t / 60, "min")

import numpy as np
from numpy.random import shuffle
#from data.data_set import DataSet

data = np.genfromtxt("../../data/mnist_seven.csv", delimiter=",", dtype="uint8")

print data[:5]
print data[:5, 0]

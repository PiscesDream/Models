import matplotlib.pyplot as plt
import matplotlib
from numpy import *
import sys

DataSize = 15
DataDim = 2
DataRange = 1000

if __name__ == '__main__':
    DataSize = int(sys.argv[1])

    data = random.rand(DataSize, DataDim) * 100
    save('data', data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[:,0], data[:,1], color='red', marker='x', ms=10, ls=' ')
    plt.show()

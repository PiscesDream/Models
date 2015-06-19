import matplotlib.pyplot as plt
from numpy import * 

def f(a, b):
    return a * log( a/b ) + (1-a) * log((1-a)/(1-b))

if __name__ == '__main__':
    x = linspace(0, 1, 1000)
    plt.plot(x, f(1e-10, x))
    plt.show()

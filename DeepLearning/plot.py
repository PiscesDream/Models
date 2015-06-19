import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcm
from pickle import load
from numpy.random import randint 
from numpy import *

def plot_dig(data, ans, ax, W = None, H = None):
    if W == None:
        W = int(len(data) ** .5)
        H = W
    ax.imshow(data.reshape(H,W), interpolation='none', vmin = 0, vmax = 1, cmap = lcm(['white','black']))        
    ax.set_title(str(ans))
#    plt.draw()

def rand_plot(xs, ys, row = 5, col = 5):
    fig, axs = plt.subplots(row, col, sharex = True, sharey = True)
    for i in xrange(row):
        for j in xrange(col):     
            ind = randint(0, xs.shape[0])
            plot_dig(xs[ind], ys[ind], axs[i,j])
    plt.show()

def all_plot(xs, ys, row = None, col = None):
    if ys == None:
        ys = ['None'] * len(xs)
    if row == None:
        l = int(len(xs)**0.5)
        fig, axs = plt.subplots(l, l, sharex = True, sharey = True)
        for i in xrange(l):
            for j in xrange(l):
                ind = i*l + j
                plot_dig(xs[ind], ys[ind], axs[i,j])
    else:
        fig, axs = plt.subplots(row, col, sharex = True, sharey = True)
        for i in xrange(row):
            for j in xrange(col):
                ind = i*col + j
                plot_dig(xs[ind], ys[ind], axs[i,j])
    plt.show()


import scipy.io as sio
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle
import gzip

def plot_dig(data, ax, W = None, H = None):
    if W == None:
        W = int(len(data) ** .5)
        H = W
    ax.imshow(data.reshape(H,W), interpolation='none', cmap = cm.gray)       

def rand_plot(data, row = 5, col = 5):
    fig, axs = plt.subplots(row, col, sharex = True, sharey = True)
    for i in xrange(row):
        for j in xrange(col):     
            ind = random.randint(0, data.shape[0])
            plot_dig(data[ind, :], axs[i,j])
    plt.show()
    
def all_plot(data, row = None, col = None):
    l = int(len(data)**0.5)
    fig, axs = plt.subplots(l, l, sharex = True, sharey = True)
    for i in xrange(l):
        for j in xrange(l):
            ind = i*l + j
            plot_dig(data[ind], axs[i,j])
    plt.show()


def load_data(num = None, threshold = None):
#===========================loading=========================
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    train_img, train_labels = train_set
    test_img, test_labels = test_set
    train_img = train_img.T
    test_img = test_img.T
#    train_labels[train_labels==0] = 10
#    test_labels[test_labels==0] = 10


    if threshold != None:
        ind = test_labels<=threshold
        test_labels = test_labels[ind]
        test_img = test_img[:, ind]
        ind = train_labels<=threshold
        train_labels = train_labels[ind]
        train_img = train_img[:, ind]


    if num != None:
        train_img = train_img[:, :num]
        train_labels = train_labels[:num]
#        test_img = test_img[:, :int(num * 0.6)]        
#        test_labels = test_labels[:int(num * 0.6)]        
        

    rand_plot(train_img.T, 3, 3)        
#    all_plot(samples)

    return train_img, train_labels, test_img, test_labels

if __name__ == '__main__':
    load_data()

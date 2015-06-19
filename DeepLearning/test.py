import autoencoder
import pickle 
import ann
from numpy import *
from plot import * 

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = pickle.load(open('digital.dat','rb'))
    print train_x.shape
    print test_x.shape
    
    cl = autoencoder.AutoEncoder([15], lmbd = 1.0, beta = 0, sparsity = 0.05)
    cl2 = ann.NeuralNetworkClassifier([100])
    print 'training...'
    cl.fit(train_x, train_x)
#    cl2.fit(train_x, train_y)
    print 'training done'
    
    print cl._f(cl.result.x)[0]  

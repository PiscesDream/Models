'''
    cnn structure copied from mnist - 08/23/14
'''
import cPickle
import gzip
import os
import sys
import time

import numpy
from numpy import *

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import time

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

class conv_pool_layer(object):
    
    def __init__(self, rng, input, filter_shape, image_shape, W = None, b = None, poolsize=(2, 2)):
        
        if W is None:
            #calc the W_bound and init the W
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                       numpy.prod(poolsize))
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype = theano.config.floatX),
                            borrow = True)
        else:
            self.W = W

        if b is None:
            b_value = numpy.zeros((filter_shape[0],), 
                                   dtype = theano.config.floatX)
            self.b = theano.shared(value = b_value, borrow = True)
        else:
            self.b = b


        conv_out = conv.conv2d(input = input, filters = self.W, 
                filter_shape = filter_shape, image_shape = image_shape)
        pooled_out = downsample.max_pool_2d(input = conv_out,
                                ds = poolsize, ignore_border = True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))

        self.params = [self.W, self.b]

def raw_dump(cnn, filename):
    import cPickle
    params = []
    for ele in cnn.cnn_layers+cnn.hid_layers:
        params.append(ele.W.get_value())
        params.append(ele.b.get_value())
    cPickle.dump((cnn.dim_in, cnn.size_in, cnn.size_out, cnn.lmbd, cnn.nkerns, cnn.nhiddens[:-1], cnn.time, params), open(filename, 'wb'))

def raw_load(filename):
    import cPickle
    dim_in, size_in, size_out, lmbd, nkerns, nhiddens, time, params = cPickle.load(open(filename, 'rb'))
    ncnn = cnn(dim_in, size_in, size_out, lmbd, nkerns, nhiddens)
    ncnn.time = time
    ind = 0
#    print len(params)
#    print len(ncnn.cnn_layers+ncnn.hid_layers)
#    print ncnn.hid_layers
#    print ncnn.cnn_layers
    for ele in ncnn.cnn_layers+ncnn.hid_layers:
        ele.W.set_value(params[ind])
        ele.b.set_value(params[ind+1])
        ind += 2
    return ncnn

class cnn(object):

    def __init__(self, dim_in, size_in, size_out, lmbd = 0.01, 
                 nkerns = [(20, (8, 8), (2, 2))], nhiddens = [10]): 
                 
        x = T.tensor4('x')
        y = T.matrix('y')
        lr = T.dscalar('lr')
        i_shp = array(size_in)
        rng = random.RandomState(10)

        #building model
        print '..building the model'

        #building the cnn
        cnn_layers = []
        for ind, ele in enumerate(nkerns):
            if ind == 0:#can opt here
                layer = conv_pool_layer(rng,
                                        input = x,
                                        image_shape = None,
                                        filter_shape = (ele[0], dim_in)+ele[1],
                                        poolsize = ele[2])
            else:
                layer = conv_pool_layer(rng,
                                        input = cnn_layers[-1].output,
                                        image_shape = None,
                                        filter_shape = (ele[0], nkerns[ind-1][0])+ele[1],
                                        poolsize = ele[2])
            i_shp = (i_shp - array(ele[1]) + 1)/array(ele[2])
            cnn_layers.append(layer)
                
        #raw_input('pause')
        
        #building the hidden
        hid_layers = []
        nhiddens = nhiddens + [size_out]
        L1 = .0
        L2 = .0
        for ind, ele in enumerate(nhiddens):
            if ind == 0:#can opt here
                layer = HiddenLayer(rng, input = T.flatten(cnn_layers[-1].output, 2),
                                    n_in = nkerns[-1][0] * i_shp.prod(),
                                    n_out = ele, activation = T.nnet.sigmoid)
            else:
                layer = HiddenLayer(rng, input = hid_layers[-1].output,
                                    n_in = nhiddens[ind-1], 
                                    n_out = ele, activation = T.nnet.sigmoid)
            L1 += abs(layer.W).sum()
            L2 += (layer.W ** 2).sum()
            hid_layers.append(layer)

        #build cost
        diff = ((y - hid_layers[-1].output) ** 2).sum()
        cost = diff + L2 * lmbd

        errors = T.mean(T.neq(T.argmax(y, 1), T.argmax(hid_layers[-1].output, 1)))

        #build update
        params = []
        for ind, ele in enumerate(cnn_layers + hid_layers):
            params.extend(ele.params)
        grads = T.grad(cost, params)

        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - lr * grad_i))

        #setting self
        self.x = x
        self.y = y
        self.cost = cost
        self.errors = errors 
        self.updates = updates
        self.cnn_layers = cnn_layers
        self.hid_layers = hid_layers
        self.nkerns = nkerns
        self.nhiddens = nhiddens
        self.lr = lr 
        self.time = [] 

        self.dim_in = dim_in
        self.size_in = size_in
        self.size_out = size_out
        self.lmbd = lmbd

    def test_error(self, x, y):
        return theano.function([], self.errors,
                               givens={self.x : x, self.y : y})();

    def fit(self, datasets, batch_size = 500, n_epochs = 200, learning_rate = 0.01):
        index = T.lscalar()

        train_set_x, train_set_y= datasets[0]
        test_set_x, test_set_y= datasets[1]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_test_batches /= batch_size

        train_model = theano.function([index], self.cost, 
            updates = self.updates,
            givens = {
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size],
                self.lr: learning_rate})
         
        test_model = theano.function([], self.errors,
            givens = {
                self.x: test_set_x,
                self.y: test_set_y})

        debug_f = theano.function([index], self.errors,
            givens = {
                self.x: test_set_x[index * batch_size : (index+1) * batch_size],
                self.y: test_set_y[index * batch_size : (index+1) * batch_size]})

#        print numpy.mean([debug_f(i) for i in xrange(n_test_batches)]) 
        raw_input(test_model())

        print '...training'
        maxiter = n_epochs
        iteration = 0
        while iteration < maxiter:
            start_time = time.time()
            iteration += 1
            print 'iteration %d' % iteration
            for minibatch_index in xrange(n_train_batches):
                print '\tL of (%03d/%03d) = %f\r' % (minibatch_index, n_train_batches, train_model(minibatch_index)),
            print ''
            print 'error = %f' % test_model()
            self.time.append(time.time()-start_time)

    def __repr__(self):
        return '<CNN: %r; HID: %r>' % (self.nkerns, self.nhiddens)

    def pred(self, x):
        return theano.function([], T.argmax(self.hid_layers[-1].output, 1), 
                        givens = {self.x: x})()

    def prob(self, x):
        return theano.function([], self.hid_layers[-1].output,
                        givens = {self.x: x})() 

def boarden(size, y):
    new = zeros((y.shape[0], size))
    for ind, ele in enumerate(y):
        new[ind][ele] = 1
    return new   

def load_cifar_data():

    print 'Loading cifar data...'
    #image_size, train_x, train_y, test_x, test_y = cPickle.load(open('../../Data/lfw/data.dat','rb')) 
    train_x, train_y, train_info, test_x, test_y, test_info = cPickle.load(open('../../Data/cifar-10/cifar-10-batches-py/full.dat','rb')) 
#   l = train_x.shape[0]
#   ind = numpy.random.permutation(l)
#   train_x = train_x[ind]
#   train_y = train_y[ind]
    train_x = train_x.reshape(-1,3,32,32)
    test_x = test_x.reshape(-1,3,32,32)
    print 'Train_x:', train_x.shape
    print 'Train_y:', train_y.shape
    print 'Test_x:', test_x.shape
    print 'Test_y:', test_y.shape

    def get_shared(data_x, data_y, borrow = True):       
        data_y = boarden(10, data_y)
        print data_y.shape
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'float64')
 
    datasets = [get_shared(train_x, train_y), get_shared(test_x, test_y)]
    print 'Loading done'

    return datasets


if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.on_unused_input='ignore'
    
    datasets = load_cifar_data()

    simple_cnn = 1 
    if simple_cnn:
        k = cnn(dim_in = 3, size_in = (32, 32), size_out = 10, 
               nkerns = [(20, (5, 5), (2, 2))], 
               nhiddens=[30])
        k.fit(datasets, n_epochs = 10, batch_size = 100, learning_rate = 0.001)

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

from mlp import HiddenLayer

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class conv_pool_layer(object):
    
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        #assert image_shape[1] == filter_shape[1]

        #calc the W_bound and init the W
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype = theano.config.floatX),
                        borrow = True)

        b_value = numpy.zeros((filter_shape[0],), 
                              dtype = theano.config.floatX)
        self.b = theano.shared(value = b_value, borrow = True)


        conv_out = conv.conv2d(input = input, filters = self.W, 
                filter_shape = filter_shape, image_shape = image_shape)
        pooled_out = downsample.max_pool_2d(input = conv_out,
                                ds = poolsize, ignore_border = True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))

        self.params = [self.W, self.b]


def cnn(size_in, size_out, datasets, lmbd = 0.01, learning_rate = 0.001, n_epochs = 200,
                nkerns = [(20, (8, 8), (2, 2))], hiddens = [10], batch_size = 500, layer0 = 1):
    #nkerns = (#feature map, filter_size, pool_size)
    #we assume that all images are in square shape


    train_set_x, train_set_y= datasets[0]
    test_set_x, test_set_y= datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size

#    raw_input( theano.function([], train_set_x.shape)())


    index = T.lscalar()
    x = T.tensor4('x')
    y = T.matrix('y')

    i_shp = array(size_in)

    rng = random.RandomState(10)

    #building model
    print '..building the model'

    layers = []
    for ind, ele in enumerate(nkerns):
        if ind == 0:
            layer = conv_pool_layer( rng, 
                    input = x,#.reshape((batch_size, layer0, i_shp, i_shp)), 
                    image_shape = None, #(batch_size, layer0, i_shp, i_shp), 
                    filter_shape = (ele[0], layer0)+ele[1], 
                    poolsize = ele[2])
            i_shp = (i_shp - array(ele[1]) + 1) / array(ele[2])
            layers.append(layer)
        else:
            layer = conv_pool_layer(rng, input = layers[-1].output,
                        image_shape = None, #(batch_size, nkerns[ind-1][0], i_shp, i_shp),
                        filter_shape = (ele[0], nkerns[ind-1][0])+ele[1], poolsize = ele[2])
            i_shp = (i_shp - array(ele[1]) + 1) / array(ele[2])
            layers.append(layer)

    #raw_input('pause')

#build the judge layer
    hid_input = T.flatten(layers[-1].output, 2)
    hid_layers = []
    hiddens = hiddens + [size_out]
    L1 = .0
    L2 = .0
    for ind, ele in enumerate( hiddens):
        if ind == 0:
            hid_layers.append(\
                HiddenLayer(rng, input = hid_input, \
                            n_in = nkerns[-1][0] * i_shp.prod(), \
                            n_out = ele, activation = T.nnet.sigmoid))
        else:
            hid_layers.append(\
                HiddenLayer(rng, input = hid_layers[-1].output,\
                            n_in = hiddens[ind-1],\
                            n_out = ele, activation = T.nnet.sigmoid))
        L1 += abs(layers[-1].W).sum() 
        L2 += (layers[-1].W ** 2).sum()

    diff = ((y - hid_layers[-1].output) ** 2).sum()
    cost = diff + L2 * lmbd

    errors = T.mean(T.neq(T.argmax(y, 1), T.argmax(hid_layers[-1].output, 1)))

    test_model = theano.function([], errors,
        givens = {
            x: test_set_x,
            y: test_set_y})

    debug_f = theano.function([index], errors,
        givens = {
            x: test_set_x[index * batch_size : (index+1) * batch_size],
            y: test_set_y[index * batch_size : (index+1) * batch_size]})

#    print numpy.mean([debug_f(i) for i in xrange(n_test_batches)])
    raw_input(test_model())
    

#update
    params = []
    for ind, ele in enumerate(layers + hid_layers):
        params.extend(ele.params)
    grads = T.grad(cost, params)
 
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, 
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})
     

    print '...training'
    maxiter = n_epochs
    iteration = 0
    while iteration < maxiter:
        iteration += 1
        print 'iteration %d' % iteration
        for minibatch_index in xrange(n_train_batches):
            print '\tL of (%03d/%03d) = %f\r' % (minibatch_index, n_train_batches, train_model(minibatch_index)),
        print ''
        print 'error = %f' % test_model()

#        if iteration % 10 == 0:
#            print '\ttraining fully model now'
#            maxiter_ = 5
#            iteration_ = 0
#            while iteration_ < maxiter_:
#                iteration_ += 1
#                print '\titeration_ %d' % iteration_
#                print '\t\tcost = %f' % numpy.mean([test_model(i) for i in xrange(n_test_batches)])

def boarden(size, y):
    new = zeros((y.shape[0], size))
    for ind, ele in enumerate(y):
        new[ind][ele] = 1
    return new
        

def load_data(dataset, num = None):
    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True, num = None):
        data_x, data_y = data_xy
        if num:
            data_x = data_x[:num]
            data_y = data_y[:num]
        data_y = boarden(10, data_y)

        size = int(data_x.shape[1]**.5)
        data_x = data_x.reshape(data_x.shape[0], 1, size, size) 
        print data_x.shape, data_y.shape

        shared_x = theano.shared(asarray(data_x,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(asarray(data_y,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set, num = num)
    valid_set_x, valid_set_y = shared_dataset(valid_set, num = num)
    train_set_x, train_set_y = shared_dataset(train_set, num = num)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.on_unused_input='ignore'
    size = 28

    datasets = load_data('../../Data/mnist/mnist.pkl.gz')
    
    cnn((28, 28), 10, datasets, n_epochs = inf, batch_size = 100,
        nkerns = [(10, (8, 4), (2, 3)), (5, (3, 2), (3, 2))], hiddens=[10],
        learning_rate = 0.001)

#    cPickle.dump([ele.get_value() for ele in classifier.ann_str.para], open('mnist_weight.dat','wb'))
#best t = 0.048100
#error = 0.032900

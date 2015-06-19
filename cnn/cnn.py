import cPickle
import gzip
import os
import sys
import time

import numpy

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

        assert image_shape[1] == filter_shape[1]

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


def cnn(size_in, size_out, lmbd = 0.01, learning_rate = 0.001, n_epochs = 200,
                nkerns = [(50, 31, 2), (30, 21, 2)], hiddens = [200], batch_size = 500, layer0 = 1):
#                nkerns = [(50, 7, 2), (30, 5, 2)],  batch_size = 50):
    #nkerns = (#feature map, filter_size, pool_size)
    #we assume that all images are in square shape

#   loaddata

#    train_set_x, train_set_y, train_set_y_0 = datasets[0]
#    test_set_x, test_set_y, test_set_y_0 = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size


    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')

    i_shp = size_in

    #building model
    print '..building the model'

    layers = []
    for ind, ele in enumerate(nkerns):
        if ind == 0:
            layer = conv_pool_layer( rng, 
                    input = x,#.reshape((batch_size, layer0, i_shp, i_shp)), 
                    image_shape = (batch_size, layer0, i_shp, i_shp), 
                    filter_shape = (ele[0], layer0, ele[1], ele[1]), 
                    poolsize = (ele[2], ele[2]))
            i_shp = (i_shp - ele[1] + 1) / ele[2]
            layers.append(layer)

        else:
            layer = conv_pool_layer(rng, input = layers[-1].output,
                        image_shape = (batch_size, nkerns[ind-1][0], i_shp, i_shp),
                        filter_shape = (ele[0], nkerns[ind-1][0], ele[1], ele[1]), poolsize = (ele[2], ele[2]))
            i_shp = (i_shp - ele[1] + 1) / ele[2]
            layers.append(layer)

    #raw_input('pause')

#build the judge layer
    hid_layers = []
    hiddens = hiddens + [size_out]
    L1 = .0
    L2 = .0
    for ind, ele in hiddens:
        if ind == 0:
            hid_layers.append(\
                HiddenLayer(input = T.flatten(layers[-1].output), \
                            n_in = nkerns[-1][0] * i_shp * i_shp, \
                            n_out = ele))
        else:
            hid_layers.append(\
                HiddenLayer(input = T.flatten(hid_layers[-1].output),\
                            n_in = hid_layers[ind-1],\
                            n_out = ele))
        L1 += abs(self.layers[-1].W).sum() 
        L2 += (self.layers[-1].W ** 2).sum()

    diff = (y - hid_layers[-1].output) ** 2
    cost = diff + L2 * lmbd

    errors = T.mean(T.neq(T.argmax(y), T.argmax(hid_layers[-1].output)))

    test_model = theano.function([index], errors,
        givens = {
            x: test_set_x[index * batch_size : (index+1) * batch_size],
            y: test_set_y_0[index * batch_size : (index+1) * batch_size]})

#update
    params = []
    for ind, ele in enumerate(layers + hid_layers):
        params.extend(ele.params)
    grads = T.grad(L, params)
 
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, 
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y_0[index * batch_size: (index + 1) * batch_size]})
     

    print '...training'
    maxiter = 4000
    iteration = 0
    while iteration < maxiter:
        iteration += 1
        print 'iteration %d' % iteration
        for minibatch_index in xrange(n_train_batches):
            print '\tL = %f' % train_model(minibatch_index)

        if iteration % 10 == 0:
            print '\ttraining fully model now'
            maxiter_ = 5
            iteration_ = 0
            while iteration_ < maxiter_:
                iteration_ += 1
                print '\titeration_ %d' % iteration_
                print '\t\tcost = %f' % numpy.mean([test_model(i) for i in xrange(n_test_batches)])
                #print theano.function([index], y, givens = {y:test_set_y_0[index * batch_size:(index+1) * batch_size]})(ind)

def load_data():

    print 'loading...'
    #image_size, train_x, train_y, test_x, test_y = cPickle.load(open('../../Data/lfw/data.dat','rb')) 
    image_size, train_x, train_y, test_x, test_y = cPickle.load(open('../../Data/mnist/data.dat','rb')) 
    l = train_x.shape[0]
    ind = numpy.random.permutation(l)
    print train_x.shape
    train_x = train_x[ind]
    train_y = train_y[ind]
    train_y_0 = (train_y+1)/2
    test_y_0 = (test_y+1)/2
    import Image
    Image.fromarray(256*train_x[0].reshape(-1, image_size)).show()

    def get_shared(data_x, data_y, data_y_0, borrow = True):       
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y.flatten(),
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y_0 = theano.shared(numpy.asarray(data_y_0.flatten(),
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y_0, 'int32')

    print 'loading done'
    datasets = [get_shared(train_x, train_y, train_y_0), get_shared(test_x, test_y, test_y_0)]

    return image_size, datasets


if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    cl = cnn()
    cl.fit()

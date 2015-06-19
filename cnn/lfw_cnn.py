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

from logistic_sgd import LogisticRegression, load_data
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
    
    def __init__(self, rng, input_A, input_B, filter_shape, image_shape, poolsize=(2, 2)):

        print image_shape
        print filter_shape
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


        conv_out_A = conv.conv2d(input = input_A, filters = self.W, 
                filter_shape = filter_shape, image_shape = image_shape)
        conv_out_B = conv.conv2d(input = input_B, filters = self.W, 
                filter_shape = filter_shape, image_shape = image_shape)
        pooled_out_A = downsample.max_pool_2d(input = conv_out_A,
                                ds = poolsize, ignore_border = True)
        pooled_out_B = downsample.max_pool_2d(input = conv_out_B,
                                ds = poolsize, ignore_border = True)


        self.output_A = T.tanh(pooled_out_A + self.b.dimshuffle('x',0,'x','x'))
        self.output_B = T.tanh(pooled_out_B + self.b.dimshuffle('x',0,'x','x'))

        self.params = [self.W, self.b]


def pyramid_cnn(learning_rate = 0.0005, n_epochs = 200,
                nkerns = [(50, 31, 2), (30, 21, 2)],  batch_size = 500):
#                nkerns = [(50, 7, 2), (30, 5, 2)],  batch_size = 50):
    #nkerns = (#feature map, filter_size, pool_size)
    #we assume that all images are in square shape
    rng = numpy.random.RandomState(123)
    
    image_size, datasets = load_data()
    print image_size

    train_set_x, train_set_y, train_set_y_0 = datasets[0]
    test_set_x, test_set_y, test_set_y_0 = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size


    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    ishape = (image_size, image_size)
    i_shp = image_size

    #building model
    print '..building the model'

    layers = []
    for ind, ele in enumerate(nkerns):
        if ind == 0:
            layer =  conv_pool_layer( rng, 
                    input_A = (x[:,:image_size * image_size]).reshape((batch_size, 1, i_shp, i_shp)), 
                    input_B = (x[:,image_size * image_size:]).reshape((batch_size, 1, i_shp, i_shp)),
                    image_shape = (batch_size, 1, i_shp, i_shp), 
                    filter_shape = (ele[0], 1, ele[1], ele[1]), 
                    poolsize = (ele[2], ele[2]))
            i_shp = (i_shp - ele[1] + 1) / ele[2]
            layers.append(layer)

        else:
            layer = conv_pool_layer(rng, input_A = layers[-1].output_A, input_B = layers[-1].output_B,
                  image_shape = (batch_size, nkerns[ind-1][0], i_shp, i_shp),
                  filter_shape = (ele[0], nkerns[ind-1][0], ele[1], ele[1]), poolsize = (ele[2], ele[2]))
            i_shp = (i_shp - ele[1] + 1) / ele[2]
            layers.append(layer)

    alpha = theano.shared(numpy.float64(.01))
    beta = theano.shared(numpy.float64(.01))
   # D = alpha * T.sum((layers[-1].output_A.flatten(2) - layers[-1].output_B.flatten(2))**2, 1) - beta
    D = T.sum((layers[-1].output_A.flatten(2) - layers[-1].output_B.flatten(2))**2, 1)
    L = T.sum(T.log(1+T.exp(y.flatten() * D)))
    #raw_input('pause')

    cnn_params = []
    for ind, ele in enumerate(layers):
        cnn_params.extend(ele.params)
    params = cnn_params #+ [alpha, beta]
    grads = T.grad(L, params)
 
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    get_delta = theano.function([index], T.mean(T.sum( (layers[-1].output_A.flatten(2) - layers[-1].output_B.flatten(2))**2, 1)),
        givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size]})

    get_D = theano.function([index], T.mean(D),
            givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size]})

    train_model = theano.function([index], L, updates=updates, 
        givens = 
            {x: train_set_x[index * batch_size: (index + 1) * batch_size],
             y: train_set_y[index * batch_size: (index + 1) * batch_size]} )

#build the judge layer
    hid_input = T.concatenate( [layers[-1].output_A.flatten(2), layers[-1].output_B.flatten(2)], 1)
    hid_l = HiddenLayer(rng, hid_input,  n_in = nkerns[-1][0] * i_shp * i_shp * 2, n_out = 200)
    log_l = LogisticRegression(input=hid_l.output, n_in = 200, n_out = 2)
    log_cost = log_l.negative_log_likelihood(y)

    test_model = theano.function([index], log_l.errors(y),
        givens = {
            x: test_set_x[index * batch_size : (index+1) * batch_size],
            y: test_set_y_0[index * batch_size : (index+1) * batch_size]})
#debug
    debug1 = theano.function([index], T.mean(log_l.p_y_given_x,0),
        givens = {
            x: test_set_x[index * batch_size : (index+1) * batch_size]})

    log_params = hid_l.params + log_l.params + cnn_params
    log_grads = T.grad(log_cost, log_params)
    log_updates = []
    for param_i, grad_i in zip(log_params, log_grads):
        log_updates.append((param_i, param_i - learning_rate *10 * grad_i))
    train_log_model = theano.function([index], log_cost, 
        updates = log_updates,
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
            print '\t\tepoch %d\%d' % (minibatch_index, n_train_batches),
            print '\tE = ' , get_delta(minibatch_index),
            print '\tD = ' , get_D(minibatch_index),
            print '\tL = %f' % train_model(minibatch_index)

        if iteration % 3 == 0:
            print '\ttraining fully model now'
            maxiter_ = 5
            iteration_ = 0
            while iteration_ < maxiter_:
                iteration_ += 1
                print '\titeration_ %d' % iteration_
                print '\t\tcost = %f' % numpy.mean([train_log_model(i) for i in xrange(n_train_batches)])
                print '\t\terror = %f' % numpy.mean([test_model(i) for i in xrange(n_test_batches)])
                ind = numpy.random.randint(0, n_test_batches)
                print ind
                print debug1(ind)
                print theano.function([index], y, givens = {y:test_set_y_0[index * batch_size:(index+1) * batch_size]})(ind)

def load_data():

    print 'loading...'
    #image_size, train_x, train_y, test_x, test_y = cPickle.load(open('../../Data/lfw/data.dat','rb')) 
    image_size, train_x, train_y, test_x, test_y = cPickle.load(open('../../Data/mnist/lfw_liked_data.dat','rb')) 
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
    #pyramid_cnn()
    pyramid_cnn(nkerns=[(20,5,2),(20,5,2)])

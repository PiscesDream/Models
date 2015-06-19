'''
    140819:
        move lr into fit
        add time recording
        add pred
        add prob
    2014/09/03:
        change the last layer into softmax
    2015/03/26:
        add the ReLU layer
        fix the no cnn bug
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
from theano.tensor.nnet import conv, softmax

import time

class conv_pool_layer(object):
    
    def __init__(self, rng, input, filter_shape, image_shape, W = None, b = None, poolsize=(2, 2),
                activation=T.tanh):
        
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
        if filter_shape == (1, 1):
            pooled_out = conv_out
        else:
            pooled_out = downsample.max_pool_2d(input = conv_out,
                                    ds = poolsize, ignore_border = True)

        self.output = activation(pooled_out + self.b.dimshuffle('x',0,'x','x'))

        self.params = [self.W, self.b]


class CNN(object):

    def __init__(self, dim_in, size_in, 
                 nkerns = [(20, (8, 8), (2, 2))], 
                 activation=T.tanh): 
                 
        x = T.tensor4('x')
        y = T.ivector('y')
        lr = T.scalar('lr')
        i_shp = array(size_in)
        rng = random.RandomState(10)

        #building model
        print '..building the cnn model: %r' % (nkerns)

        #building the cnn
        cnn_layers = []
        if nkerns!=[]:
            for ind, ele in enumerate(nkerns):
                if ind == 0:#can opt here
                    input = x
                    filter_shape = (ele[0], dim_in)+ele[1]
                else:
                    input = cnn_layers[-1].output
                    filter_shape = (ele[0], nkerns[ind-1][0]) + ele[1]

                layer = conv_pool_layer(rng,
                                        input = input,
                                        image_shape = None,
                                        filter_shape = filter_shape ,
                                        poolsize = ele[2],
                                        activation = activation) 

                cnn_layers.append(layer)
                i_shp = (i_shp - array(ele[1]) + 1)/array(ele[2])

        output = T.flatten(cnn_layers[-1].output if nkerns!=[] else x, 2) 
        #raw_input('pause')

        index = T.lscalar()

        self.index = index
        self.x = x
        self.y = y
        self.output = output
        self.i_shp = i_shp
        self.cnn_layers = cnn_layers
        self.nkerns = nkerns
        self.lr = lr

    def get_feature(self, x):
        return theano.function([], self.output, givens={self.x:x})()

    def set_lossf(self, lossf):
        try:
            loss = lossf(self)
        except:
            loss = lossf
        #loss, debug = lossf(self)
        params = []
        for ele in self.cnn_layers:
            params.extend(ele.params)
        grads = T.grad(loss, params)

        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - self.lr * grad_i))

        self.loss = loss
        self.updates = updates



    def fit(self, datasets, batch_size = 500, n_epochs = 200, learning_rate=0.01,
            test_model_on = True):
#       building loss function
        print 'building loss function...'
        
        index = self.index

        print 'building training model...'

        train_set_x, train_set_y= datasets[0]
        test_set_x, test_set_y= datasets[1]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_test_batches /= batch_size

        train_model = theano.function([index], self.loss, 
            updates = self.updates,
            givens = {
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size],
                self.lr:learning_rate})
        
        if test_model_on:
            test_model = theano.function([], self.loss,
                givens = {
                    self.x: test_set_x,
                    self.y: test_set_y})


#       debug_model = theano.function([index], self.output.shape, 
#           updates = self.updates,
#           givens = {
#               self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
#               self.y: train_set_y[index * batch_size: (index + 1) * batch_size],
#               self.lr: learning_rate})
#       print debug_model(0)
#       raw_input('debug')


        #raw_input(test_model())
        if test_model_on: print test_model()

        print 'training...'
        maxiter = n_epochs
        iteration = 0
        while iteration < maxiter:
#            start_time = time.time()
            iteration += 1
            print 'iteration %d' % iteration
            for minibatch_index in xrange(n_train_batches):
                print '\tL of (%03d/%03d) = %f\r'%(minibatch_index, n_train_batches, train_model(minibatch_index)),
#                print '\tL of (%03d/%03d) = %f'%(minibatch_index, n_train_batches, train_model(minibatch_index))
            print ''
            if test_model_on: print 'error = %f' % test_model()
#            self.time.append(time.time()-start_time)



    def __repr__(self):
        return '<Pure CNN: %r>' % (self.nkerns, self.nhiddens)



def load_data(dataset, num = None):
    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    all_x = concatenate([train_set[0], valid_set[0], test_set[0]], 0)
    all_y = concatenate([train_set[1], valid_set[1], test_set[1]], 0)
    total_size = all_x.shape[0]

    random_index = numpy.random.permutation(total_size)
    all_x = all_x[random_index]
    all_y = all_y[random_index]

    sep = int(total_size * 0.7)
    test_set = (all_x[sep:], all_y[sep:])
    train_set = (all_x[:sep], all_y[:sep])


    def shared_dataset(data_xy, borrow=True, num = None):
        data_x, data_y = data_xy
        if num:
            data_x = data_x[:num]
            data_y = data_y[:num]

        size = int(data_x.shape[1]**.5)
        data_x = data_x.reshape(data_x.shape[0], 1, size, size) 
        print data_x.shape, data_y.shape

        shared_x = theano.shared(asarray(data_x,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(asarray(data_y,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set, num = num)
#    valid_set_x, valid_set_y = shared_dataset(valid_set, num = num)
    train_set_x, train_set_y = shared_dataset(train_set, num = num)

    rval = [(train_set_x, train_set_y), #(valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import loss_functions 

    theano.config.exception_verbosity='high'
    theano.config.on_unused_input='ignore'
    size = 28

    datasets = load_data('../../../Data/mnist/mnist.pkl.gz', 1000)
    datasets_for_abstract = load_data('../../../Data/mnist/mnist.pkl.gz', 10000)
 
    def examinate(cnn):
        x = datasets[0][0].eval()
        y = datasets[0][1].eval()
        k = cnn.nkerns[-1][0]
        shp = cnn.i_shp
        for i in range(10):
            cen = x[y==i]
            cen = cen.sum(0)/cen.shape[0]
            feature = cnn.get_feature(cen.reshape(1, 1, 28, 28))
        
            plt.subplot(2, 5, i+1)
            plt.imshow(feature.reshape(k, shp[0], shp[1])[0], interpolation='None', cmap='binary')
        plt.show()

    def dump2file(cnn, filename):
        global datasets_for_abstract
        x = datasets_for_abstract[0][0]
        y = datasets_for_abstract[0][1].eval()

        features = cnn.get_feature(x)

        print features.shape
        print y.shape
        cPickle.dump((features, y), open(filename, 'wb'))
        

    cnn = CNN(dim_in = 1, size_in = (28, 28), nkerns = [(8, (2, 2), (1, 1)), (6, (3, 3), (2, 2))])
    #fit(self, lossf, datasets, batch_size = 500, n_epochs = 200, learning_rate = 0.01):

#    examinate(cnn)
    dump2file(cnn, './features/1_random.feature')

    cnn.fit(loss_functions.lossf1, datasets, batch_size = 200, n_epochs = 500, learning_rate = 0.0001)

    dump2file(cnn, './features/1_trained.feature')
    examinate(cnn)




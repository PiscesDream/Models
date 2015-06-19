'''
    update:
        2014/09/03:
            softmax in the last layer
'''
import theano
import theano.tensor as T
import gzip
import cPickle

import numpy
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

class ANN(object):
    
    def __init__(self, n_in, n_out, lmbd = 0.01, hiddens = [10]):
       
        x = T.matrix('x')
        y = T.ivector('y')
        lr = T.scalar('lr')
        rng = numpy.random.RandomState(numpy.random.randint(2 ** 30))

        params = []
        hid_layers = []
        L2 = .0
        n_hid = hiddens + [n_out]
        for ind, ele in enumerate(n_hid):
            if ind == 0:
                input = x
                n_in = n_in
            else:
                input = hid_layers[-1].output
                n_in = n_hid[ind-1] 

            if ind == len(n_hid) - 1:
                activation = T.nnet.softmax
            else:
                activation = T.nnet.sigmoid

            layer = HiddenLayer(rng, input = input, n_in = n_in, n_out = ele, activation = activation)
            hid_layers.append( layer)

            L2 += T.sum(layer.W ** 2) 
            
            params.extend([layer.W, layer.b])

        nl = -T.mean(T.log(hid_layers[-1].output)[T.arange(y.shape[0]), y])
        cost = nl + L2 * lmbd

        grads = T.grad(cost, params)

        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - lr * grad_i))

        y_pred = T.argmax(hid_layers[-1].output, 1)
        errors = T.mean(T.neq(y_pred, y))
        
       
        self.n_in = n_in
        self.n_out = n_out
        self.hiddens = hiddens
        
        self.x = x
        self.y = y
        self.lr = lr
        
        self.cost = cost
        self.errors = errors
        self.updates = updates
        self.pred = y_pred
        self.time = []


    def fit(self, datasets, batch_size = 500, n_epochs = 200, lr = 0.01):
        ''' without validation'''

        index = T.lscalar()
        
        train_set_x, train_set_y = datasets[0]
        test_set_x, test_set_y = datasets[1]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_test_batches /= batch_size

        train_model = theano.function([index], self.cost, 
            updates = self.updates,
            givens = {
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size],
                self.lr: lr})
         
        test_model = theano.function([], self.errors,
            givens = {
                self.x: test_set_x,
                self.y: test_set_y})

        debug_f = theano.function([index], self.errors,
            givens = {
                self.x: test_set_x[index * batch_size : (index+1) * batch_size],
                self.y: test_set_y[index * batch_size : (index+1) * batch_size]})

#        print numpy.mean([debug_f(i) for i in xrange(n_test_batches)]) 
        print(test_model())

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

    def __repr__(self):
        return '<ANN:%r-%r-%r>' % (self.n_in, self.hiddens, self.n_out)



def load_data(dataset, num = None):
    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_set = (numpy.concatenate([train_set[0], valid_set[0]], 0), numpy.concatenate([train_set[1], valid_set[1]], 0))
    f.close()

    def shared_dataset(data_xy, borrow=True, num = None):
        data_x, data_y = data_xy
        if num:
            data_x = data_x[:num]
            data_y = data_y[:num]
#        data_y = boarden(10, data_y)

        size = int(data_x.shape[1]**.5)
#        data_x = data_x.reshape(data_x.shape[0], -1) 
        print data_x.shape, data_y.shape

        shared_x = theano.shared(numpy.asarray(data_x,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
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
    theano.config.exception_verbosity='high'
    theano.config.on_unused_input='ignore'

    datasets = load_data('../../Data/mnist/mnist.pkl.gz')

    cl = ANN(28 * 28, 10, hiddens = [1])
    cl.fit(datasets, lr = 0.1)











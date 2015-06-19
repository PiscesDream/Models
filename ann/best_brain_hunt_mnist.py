import theano, cPickle, gzip
import Image
import theano.tensor as T
from numpy import *
import sys
import time

class hidden_layer(object):
    
    def __init__(self, rng, input, n_in, n_out, W = None, b = None, activation = T.tanh):
        
        if W is None:
            W_values = asarray(rng.uniform(
                    low=-sqrt(6. / (n_in + n_out)),
                    high=sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W, self.b]        


class ann_struct(object):
    
    def __init__(self, input, n_in, n_out, h_layers = [100]):
        rng = random.RandomState(32)
        self.input = input
        
        ls = [n_in] + h_layers + [n_out]
        self.layers = []      
        self.L1 = .0
        self.L2 = .0
        self.para = []

        for ind in xrange(len(ls)-1):
            if ind == 0:
                self.layers.append(hidden_layer(rng, self.input, ls[0], ls[1]))
            else:
                self.layers.append(hidden_layer(rng, self.layers[-1].output, ls[ind], ls[ind+1]))
            self.L1 += abs(self.layers[-1].W).sum() 
            self.L2 += (self.layers[-1].W ** 2).sum()
            self.para.extend(self.layers[-1].params)


    def predict(self, input = None):
        if input == None:            
            return self.layers[-1].output
        else:
            return theano.function(inputs = [], outputs = self.layers[-1].output, givens = {self.input:input})()             

#    def error(self, y):
#        return T.mean(T.neq(T.argmax(self.predict().T) - y))
    def error(self, y):
#        return T.mean(T.neq(T.sgn(self.predict()), y))
        return T.mean(T.neq(T.argmax(self.predict(),1 ), T.argmax(y, 1)))


    def dist(self, y):
        return ((self.predict() - y) ** 2).sum()

    def __repr__(self):
        return repr(map(lambda x: x.get_value().shape, self.para)) + repr(self.ls)


class ann(object):

    def __init__(self, n_in, n_out, h_layers = [10], lmbd1 = 0, lmbd2 = 0.0001, learning_rate = 0.01, n_epochs=500):
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        self.n_epochs = n_epochs

        self.ann_str = ann_struct(self.x, n_in, n_out, h_layers)
        self.cost = self.ann_str.dist(self.y) + lmbd1 * self.ann_str.L1 + lmbd2 * self.ann_str.L2
        self.updates = map(lambda x: (x, x - learning_rate * T.grad(self.cost, x)), self.ann_str.para)


        #best brain hunt
        for ind, ele in enumerate(self.updates):
            t = ele[0].get_value()
            if t.ndim == 2:
                for x, row in enumerate(t):
                    for y, cell in enumerate(row):
                        if (abs(cell) < 1e-1):
                            updates[ind][1]

#        self.predict = theano.function(inputs = [seflx], outputs = ann_str.predict(x))

    def fit(self, datasets, batch_size = 20):

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]       
   
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar('index')

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
        
        print self.ann_str
#        print theano.function(inputs=[],outputs=T.argmax(self.ann_str.predict(), 1),givens={self.x:train_set_x})()
#        print theano.function(inputs=[],outputs=T.argmax(train_set_y, 1))()
#        print theano.function(givens={self.x:train_set_x,self.y:train_set_y}, inputs = [], outputs = self.cost)()

        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        test_model = theano.function(inputs=[index],
                outputs=self.ann_str.error(self.y),
                givens={
                    self.x: test_set_x[index * batch_size:(index + 1) * batch_size],
                    self.y: test_set_y[index * batch_size:(index + 1) * batch_size]})

        validate_model = theano.function(inputs=[index],
                outputs=self.ann_str.error(self.y),
                givens={
                    self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                    self.y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index], outputs=self.cost,
                updates=self.updates,
                givens={
                    self.x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    self.y: train_set_y[index * batch_size:(index + 1) * batch_size]})

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_params = None
        best_validation_loss = inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

#                print self.ann_str.layers[-1].W.get_value()
                minibatch_avg_cost = train_model(minibatch_index)
#                print self.ann_str.layers[-1].W.get_value()
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                         (epoch, minibatch_index + 1, n_train_batches,
                          this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                        done_looping = True
                        break

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
#        print >> sys.stderr, ('The code for file ' +
#                              os.path.split(__file__)[1] +
#                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

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
        if num != None:
            data_x = data_x[:num]
            data_y = data_y[:num]
        data_y = boarden(10, data_y)
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
    size = 28
    classifier = ann(size**2, 10, [30, 25, 20])
    
    print 'loading...'
    test_dataset = 'mnist'
    if test_dataset == 'lfw':
        train_x, train_y, test_x, test_y = cPickle.load(open('../../Data/lfw/data.dat','rb')) 
        l = train_x.shape[0]
        ind = random.permutation(l)
        train_x = train_x[ind]
        train_y = train_y[ind]
        valid_x = train_x[:int(l*0.3)]
        valid_y = train_y[:int(l*0.3)]
        train_x = train_x[int(l*.3):]
        train_y = train_y[int(l*.3):]
    elif test_dataset == 'mnist':
        f = gzip.open('../../Data/mnist/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        train_x, train_y = train_set
        valid_x, valid_y = valid_set
        test_x, test_y = test_set
        
        def process_x(data_x):
            return asarray(data_x, dtype = 'float64')
        def process_y(data_y):
            tmp = zeros((data_y.shape[0], 10))
            for ind, ele in enumerate(tmp):
                tmp[ind, data_y[ind]] = 1
            tmp[tmp==0] = -1
            return tmp
        train_x = process_x(train_x)
        valid_x = process_x(valid_x)
        test_x = process_x(test_x)  

        train_y = process_y(train_y)
        valid_y = process_y(valid_y)
        test_y = process_y(test_y)

        size = int(train_x.shape[1] ** .5) 
    Image.fromarray(train_x[0].reshape(size, -1)*256).show()


    def get_shared(data_x, data_y, borrow = True):       
        shared_x = theano.shared(asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_y

    print 'loading done'
    datasets = [get_shared(train_x, train_y), get_shared(valid_x, valid_y), get_shared(test_x, test_y)]
    classifier.fit(datasets)


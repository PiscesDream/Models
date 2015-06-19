import numpy as np
import scipy as np
from operator import le
import pso
import pso_adv

class PSO_ANN(object):

    def __init__(self, hidden_layers = None, lmbd = 1.0,
        p_size = 20, w = 0.8, c1 = 2, c2 = 2, maxiter = 2000, vmax = 1):

        if hidden_layers is None:
            self.hidden_layers = [100]
        else:
            self.hidden_layers = hidden_layers

        self.L = 2 + len(self.hidden_layers)
        self.A = [None] * self.L
        self.theta = [None] * (self.L - 1)
        self.delta = [None] * self.L
        self.grad = [None] * self.L
        self.lmbd = lmbd

        self.p_size = p_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.maxiter = maxiter
        self.vmax = vmax

    @staticmethod
    def _sigma(x):
        return 1.0 / (1 + np.exp(-x))

    def _f(self, Theta):
        self.A[0] = self.X

        theta_idx = 0

        for i in range(self.L - 1):
            theta_shape = (self.layers[i] + 1, self.layers[i + 1])
            theta_len = theta_shape[0] * theta_shape[1]
            self.theta[i] = (Theta[theta_idx : theta_idx + theta_len].
                    reshape(theta_shape))
            theta_idx += theta_len

            z = np.hstack((
                np.ones((self.A[i].shape[0], 1)),
                self.A[i])).dot(self.theta[i])
            self.A[i + 1] = self._sigma(z)

        mJ = (- (self.target * np.log(self.A[-1]) +
            (1 - self.target) * np.log(1 - self.A[-1])).sum()
            + self.lmbd * 0.5 * (Theta ** 2).sum())
        #Theta**2?

#        self.delta[-1] = self.A[-1] - self.target
#        for i in range(self.L - 2, 0, -1):
#            self.delta[i] = (self.delta[i + 1].dot(self.theta[i].T[:, 1:]) *
#                    self.A[i] * (1 - self.A[i]))

#        for i in range(self.L - 1):
#            self.grad[i] = np.vstack((
#                np.ones((1, self.A[i].shape[0])),
#                self.A[i].T)).dot(self.delta[i + 1])

#        Grad = np.concatenate(map(lambda x: x.flatten(), self.grad[:-1]))
#        Grad += self.lmbd * Theta

        return mJ

    def init_theta(self):
        init_thetas = [None] * (self.L - 1)
        for i in range(self.L - 1):
            epsilon = np.sqrt(6.0 / (self.layers[i] + self.layers[i + 1]))
            init_thetas[i] = np.random.mtrand.rand(self.layers[i] + 1,
                    self.layers[i + 1]) * 2.0 * epsilon - epsilon
        return np.concatenate(map(lambda x: x.flatten(), init_thetas))  

    def fit(self, X, y): 
        self.X = X 
        target_labels = sorted(list(set(y)))
        #cut the redundant
        labels_count = len(target_labels)
        self.labels_map = dict(zip(target_labels, range(labels_count)))
        self.labels_index_map = dict(zip(range(labels_count), target_labels))

        self.target = np.zeros((X.shape[0], labels_count))
        for i, label in enumerate(y):
            self.target[i, self.labels_map[label]] = 1

        self.layers = [X.shape[1]]
        self.layers.extend(self.hidden_layers)
        self.layers.append(labels_count)
        
#        self.result = optimize.minimize(self._f, x0 = init_theta,
#                method = self.optimization_method, jac = True,
#                options = self.method_specific_options)
        self.result = pso_adv.PSO(self._f, le, self.init_theta, 
                            self.p_size, lambda x: all(x<3) and all(x>-3), self.vmax,
                            self.w, self.c1 , self.c2, self.maxiter,
                            nor_perceptron = 10, nor_r = 0.2).get_ans()

        self.optimized_theta = []
        optimized_theta = self.result[0]
        theta_idx = 0
        for i in range(self.L - 1):
            theta_shape = (self.layers[i] + 1, self.layers[i + 1])
            theta_len = theta_shape[0] * theta_shape[1]
            self.optimized_theta.append(
                    optimized_theta[theta_idx : theta_idx + theta_len]
                    .reshape(theta_shape))
            theta_idx += theta_len

    def predict(self, X):
        labels_idx = self.predict_proba(X).argmax(axis = 1)
        return map(lambda x: self.labels_index_map[x], labels_idx)

    def predict_proba(self, X):
        self.A[0] = X
        m = X.shape[0]

        for i in range(self.L - 1):
            _X = np.hstack((np.ones((m, 1)), self.A[i]))
            self.A[i + 1] = self._sigma(_X.dot(self.optimized_theta[i]))

        return self.A[-1]


from test_fun import *
from numpy import * 
from operator import le, gt
import pso_adv
import pso
import pso_w
import matplotlib.pyplot as plt

def f(x):
    return Rastrigin(x)

def plot(xs, best_pos, best, iteration):
    a = linspace(-5.12, 5.12, 100)
    plt.clf()
    plt.plot(a, map(lambda x: f(x), a), color = 'blue')
    ys = map(lambda x: f(x), xs)
    plt.plot(xs, ys, 'ro')    
    plt.draw()

    plt.pause(0.3)

def plot2(xs, best_pos, best, iteration, w = None):
    print '\t',iteration, '\t\tf(',best_pos,') =', best, 'w =', w, '\r',

#    plt.pause(0.3)

if __name__ == '__main__':   
#    plt.plot(a, map(lambda x: Rastrigin([x]), a))    
#    plt.show()


#    plt.ion()
#    plt.show()

    dimension = 5
    fi = open('rec.txt','w')    
    for iteration in xrange(50):
        print 'test %02d:'% iteration
        fi.write('test %02d\n'%iteration)

#        init_x = (random.rand(500, dimension) - 0.5) * 10

#        print('pso_w')
#        x = pso_w.PSO(f, le,lambda:(random.rand(3) - 0.5)*10,
#            50, lambda x: all(x<5) and all(x>5), 10,
#            w = 1.2, c1 = 2, c2 = 2,
#            maxiter = 1000,
#            plot_f = plot2,
#            init_x = init_x).get_ans()
#        fi.write('\tpso_w best fitness: f('+str(x[0])+')='+str(x[1])+'\n')
        

        print('pso_ori')
        x = pso.PSO(f, le, lambda:(random.rand(dimension)-.5)*10,
            50, lambda x: all(x<5) and all(x>5), 10,           
            w = 0.8, c1 = 2, c2 = 2,
            maxiter = 1000,
            plot_f = plot2).get_ans()
        fi.write('\tpso   best fitness: f('+str(x[0])+')='+str(x[1])+'\n')

        print('pso_adv')
        x = pso_adv.PSO(f, le, lambda:(random.rand(dimension)-.5)*10,
            50, lambda x: all(x<5) and all(x>5), 10,
            w = 0.8, c1 = 2, c2 = 2,
            maxiter = 200,
            plot_f = plot2,
            gen_rate = 0.0, gen_r = 1,
            nor_perceptron = 5, nor_r = 0.1)

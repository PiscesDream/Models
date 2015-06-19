from numpy import * 

class PSO(object):
    '''particle swarm optimization
       f: fit function
       cmp_f: compare function 
       init_f: initialization function
       p_size: number of particles
       bound_f: bound function(bool)
       vmax: maximum moving velocity

       gen_rate: like GA, the best biont's child in guassian
       gen_r : guassian radium

       nor_perceptron: the number of sense point
       nor_r: the radium of sense area
       '''
    
    def __init__(self, f, cmp_f, init_f,
        p_size, bound_f, vmax,
        w = 0.8, c1 = 2, c2 = 2, maxiter = 2000,
        plot_f = None,
        gen_rate = 0.0, gen_r = 0.3,
        nor_perceptron = 0, nor_r = 0.2, init_x = None):

        pbest = array([None] * p_size)
        pbest_pos = array([None] * p_size)
        gbest = None
        gbest_pos = None
        v = list(random.rand(p_size)*vmax)
        p = []       
        if init_x == None:
            for i in xrange(p_size):
                p.append(init_f())
        else:
            p = init_x
        p = array(p)

        for iteration in xrange(maxiter):
            for ind, ele in enumerate(p):                
                best_f = f(ele)
                for i in xrange(nor_perceptron):
                    t = array(map(lambda x: random.normal(x, nor_r), ele))
                    tmp = f(t)
                    if best_f == None or cmp_f(tmp, best_f):
                        best_f = tmp
                        p[ind] = t
                        ele = t
                if pbest[ind] == None or cmp_f(best_f, pbest[ind]):
                    pbest[ind] = best_f
                    pbest_pos[ind] = ele.copy()
                    if gbest == None or cmp_f(best_f, gbest):
                        gbest = best_f
                        gbest_pos = ele.copy()
            for ind, ele in enumerate(p):
                v[ind] = w * v[ind]\
                  + c1 * random.rand() * (pbest_pos[ind] - ele)\
                  + c2 * random.rand() * (gbest_pos - ele)
                if any(v[ind] > vmax):
                    v[ind] = vmax
                p[ind] += v[ind]
                if bound_f(p[ind])==False:
                    p[ind] = init_f()
#            print 'the %dth iter:\tbest fitness: f(' % iteration, gbest_pos, ')=',gbest

            survive = int((1-gen_rate)*p_size)
            idx = pbest.argsort()
            pbest = pbest[idx]
            pbest_pos = pbest_pos[idx]
            p = p[idx]
            for ind in xrange(survive+1, len(p)):
                p[ind] = map(lambda x: random.normal(x, gen_r), gbest_pos)                

            if plot_f != None:
                plot_f(p, gbest_pos, gbest, iteration)

        print 'best fitness: f(', gbest_pos, ')=', gbest


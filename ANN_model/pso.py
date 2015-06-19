from numpy import * 

class PSO(object):
    '''particle swarm optimization
       f: fit function
       cmp_f: compare function 
       init_f: initialization function
       p_size: number of particles
       bound_f: bound function(bool)
       vmax: maximum moving velocity
       '''
    
    def __init__(self, f, cmp_f, init_f,
        p_size, bound_f, vmax,
        w = 0.8, c1 = 2, c2 = 2, maxiter = 2000,
        plot_f = None, init_x = None):

        pbest = [None] * p_size
        pbest_pos = [None] * p_size
        gbest = None
        gbest_pos = None
        v = list(random.rand(p_size)*vmax)
        p = []       
        if init_x == None:
            for i in xrange(p_size):
                p.append(init_f())
        else:
            p = init_x

        for iteration in xrange(maxiter):
            for ind, ele in enumerate(p):
                tmp = f(ele)
                if pbest[ind] == None or cmp_f(tmp, pbest[ind]):
                    pbest[ind] = tmp.copy()
                    pbest_pos[ind] = ele.copy()
                    if gbest == None or cmp_f(tmp, gbest):
                        gbest = tmp.copy()
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
            if plot_f != None:
                plot_f(p, gbest_pos, gbest, iteration)
            else:
                print iteration, '\t\tf(',gbest_pos,') =', gbest
#            print 'the %dth iter:\tbest fitness: f(' % iteration, gbest_pos, ')=',gbest

        print 'best fitness: f(', gbest_pos, ')=', gbest
        self.gbest_pos = gbest_pos
        self.gbest = gbest

    def get_ans(self):
        return self.gbest_pos, self.gbest

from numpy import * 
#from copy import deepcopy

def sigmoid(x):
    return 1.0/(1+exp(-x)) - 0.5

class PSO(object):
    '''particle swarm optimization
       f: fit function
       cmp_f: compare function 
       init_f: initialization function
       p_size: number of particles
       bound_f: bound function(bool)
       vmax: maximum moving velocity

       it's weight is changing everytime
       '''
    
    def __init__(self, f, cmp_f, init_f,
        p_size, bound_f, vmax,
        w = 1.2, c1 = 2, c2 = 2, maxiter = 2000,
        plot_f = None, init_x = None):

        _c1 = [c1] * p_size
        _c2 = [c2] * p_size

        pbest = [None] * p_size
        pbest_pos = [None] * p_size
        pbest_delta = [None] * p_size

        gbest = None
        gbest_pos = None
        gbest_delta = None

        v = list(random.rand(p_size)*vmax)
        p = []       
        if init_x == None:
            for i in xrange(p_size):
                p.append(init_f())  
        else:
            p = init_x

        k = 0.6 - w            
        ori_w = float(w)

        for iteration in xrange(maxiter):
            for ind, ele in enumerate(p):
                tmp = f(ele)
                if pbest[ind] == None or cmp_f(tmp, pbest[ind]):
                    if pbest[ind] != None:
                        pbest_delta[ind] = tmp - pbest[ind]
                    pbest[ind] = tmp.copy()
                    pbest_pos[ind] = ele.copy()

                    if gbest == None or cmp_f(tmp, gbest):
                        if gbest != None:
                            gbest_delta = tmp - gbest
                        gbest = tmp.copy()
                        gbest_pos = ele.copy()            

            #------update the weight--------
            #no good function, a waste
#            if gbest_delta != None:
#                for ind, ele in enumerate(pbest_delta):              
#                    if ele != None:
#                        _c1[ind] = c1 + (ele/gbest_delta)
#                        _c2[ind] = c2 + (gbest_delta/ele)
#                        pass


            #-------updata the w------
#            w = exp(-iteration)*(ori_w-0.6) + 0.6

#            w += k/maxiter

#            if gbest_delta != None:
#                w -= exp(-abs(gbest_delta))

            #------moving-----
            for ind, ele in enumerate(p):
                v[ind] = w * v[ind]\
                  + _c1[ind] * random.rand() * (pbest_pos[ind] - ele)\
                  + _c2[ind] * random.rand() * (gbest_pos - ele)
                if any(v[ind] > vmax):
                    v[ind] = vmax
                p[ind] += v[ind]
                if bound_f(p[ind])==False:
                    p[ind] = init_f()
            if plot_f != None:
                plot_f(p, gbest_pos, gbest, iteration, w)
#            print 'the %dth iter:\tbest fitness: f(' % iteration, gbest_pos, ')=',gbest

        print 'best fitness: f(', gbest_pos, ')=', gbest
        self.gbest_pos = gbest_pos
        self.gbest = gbest

    def get_ans(self):        
        return self.gbest_pos, self.gbest

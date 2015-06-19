'''
    rewrite in 2014/10/01:
        
        - all multiprocess
        - standard GA
'''

import multiprocessing as mp 
from numpy import *
from time import time

class extended_Queue(object):
    
    def __init__(self, max = 500, threshold = 50):
        self.qs = [mp.Queue() for i in range(max)]
        self.ind = 0
        self.max = max
        self.threshold = threshold
        self.count = 0

    def put(self, ele):
        if self.qs[self.ind].qsize() > self.threshold:
            self.ind += 1
            if self.ind >= self.max:
                pass
#                raw_input('leaking')
        self.qs[self.ind].put(ele)

    def get(self):
        if self.qs[self.ind].empty():
            self.ind -= 1
#        print self.ind
#        print self.__repr__()
        return self.qs[self.ind].get()
    
    def refresh(self):
        while self.qs[self.ind].qsize() > self.threshold:
            self.ind += 1

    def __repr__(self):
        return '<Extended Queue; Qsizes: '+str(map(lambda x: x.qsize(), self.qs)) + '>'

def choice(eles, num, p):
    ans = []
    for i in xrange(num):
        acc = .0
        thr = random.rand()
        for ind, ele in enumerate(p):
            acc += ele
            if acc > thr:
                ans.append(eles[ind])
                break
    return ans                


class GA(object):
    
    def __init__(self, fitness_f, terminator, generation_size, generation_init, 
        p_crossover, crossover_f, mutation_rate, mutation_f, 
        plot = False, plot_f = None, plot_interval = 100,
        cmp_f = None,
        cores = 1 ):

        self.fitness_f = fitness_f
        if 'fitness_thresold' in terminator:
            self.fitness_thresold = terminator['fitness_thresold']
            self.iter_maximum = None
        else:
            self.iter_maximum = terminator['iter_maximum']
            self.fitness_thresold = None

        #adjust for multiprocess
        self.generation_size = ((generation_size-1)/cores + 1) * cores
        self.generation_init = generation_init
        self.p_crossover = p_crossover 
        self.crossover_f = crossover_f
        self.mutation_rate = mutation_rate
        self.mutation_f = mutation_f
        self.plot = plot
        self.plot_f = plot_f
        self.best_fitness = None
        self.plot_interval = plot_interval
        self.cores = cores

        if cmp_f == None:
            self.cmp_f = lambda a,b: a>b
        else:
            self.cmp_f = cmp_f

    def result(self):
        self.best_fitness = None
        self.fit()
        return self.best_biont
    
    def fit(self):            
        res = extended_Queue()
        
        #mission
        def f(times, res):
            for i in xrange(times):
                biont = self.generation_init()
                fitness = self.fitness_f(biont)
                res.put((biont, fitness))
        #arrange the tasks
        tasks = []
        for ind in xrange(self.cores):
            p = mp.Process(target = f, name = 'Process: %d' % ind, args = (self.generation_size/self.cores,res))
            tasks.append(p)
            p.start()
        for task in tasks:
            task.join()
        res.refresh()

        generation = [None] * self.generation_size
        fitness = [None] * self.generation_size
        fitness_max = None

        for ind in xrange(self.generation_size):
            biont__, fitness__ = res.get()
            generation[ind] = biont__
            fitness[ind] = fitness__
            if fitness_max == None or self.cmp_f(fitness__, fitness_max):
                fitness_max = fitness__
                best_child = biont__
        fitness_sum = sum(fitness)
        
        iteration = 0
        while (self.fitness_thresold == None and iteration < self.iter_maximum) or \
            (self.iter_maximum == None and not self.cmp_f(fitness_max,self.fitness_thresold)):

            start_time = time()

            if self.best_fitness == None or self.cmp_f(fitness_max, self.best_fitness):
                self.best_fitness = fitness_max
                self.best_biont = best_child

            print '%03dth  generation|\tbest fitness:\t%lf|\tbest child fitness:\t%lf' % (iteration, self.best_fitness,fitness_max)

            if self.plot and iteration % self.plot_interval == 0:
                self.plot_f(best_child, 'red', False)            
                self.plot_f(self.best_biont, 'blue', True)

            iteration += 1
            


            #generation probability
            gen_pr = array(fitness)
            gen_pr = gen_pr - gen_pr.min()
            if gen_pr.sum() == 0: gen_pr = 1 - gen_pr
            gen_pr = gen_pr / gen_pr.max()
            gen_pr = exp(gen_pr)
            gen_pr = gen_pr / gen_pr.sum()

            #sample next generation
            next_generation = []
            seeds = sort(random.uniform(0, 1, size = (self.generation_size,)))
            acc = 0; cur = 0; ind = 0
            while ind < self.generation_size:
                acc += gen_pr[cur]
                while ind < self.generation_size and seeds[ind] <= acc:
                    next_generation.append(generation[cur])
                    ind += 1
                cur += 1



            #crossover
            seeds = list((random.binomial(1, self.p_crossover, self.generation_size)\
                     * range(self.generation_size)).nonzero()[0])
            if len(seeds) % 2 == 1:
                seeds.pop(-1)

            def f(mission, res):
                for father, mother in mission:
                    child0, child1 = self.crossover_f(father[1], mother[1])
                    res.put((father[0], child0))
                    res.put((mother[0], child1))
            
            missions = [[] for i in range(self.cores)]
            l = len(seeds)
            for ind in range(l)[::2]:
                ind0 = seeds[ind]
                ind1 = seeds[ind+1]
                missions[ind/2 % self.cores].append( ( (ind0, next_generation[ind0]),
                                                       (ind1, next_generation[ind1])  )  )
            tasks = []
            for ind, ele in enumerate(missions):
                p = mp.Process(target = f, args = (ele, res)) 
                p.start()
                tasks.append(p)
            for ele in tasks:
                ele.join()

            res.refresh()
            for ind__ in range(l):
                ind, ele = res.get()
                next_generation[ind] = ele
            print 'cross mission', map(lambda x: len(x), missions)




            #mutation
            seeds = list((random.binomial(1, self.mutation_rate, self.generation_size)\
                     * range(self.generation_size)).nonzero()[0])

            def f(mission, res):
                for ind, ele in mission:
                    res.put((ind, self.mutation_f(ele)))

            missions = [[] for i in range(self.cores)]
            for ind, ele in enumerate(seeds):
                missions[ind % self.cores].append( (ele, next_generation[ele]) )

            tasks = []
            for ind, ele in enumerate(missions):
                p = mp.Process(target = f, args=(ele, res))
                p.start()
                tasks.append(p)
            for task in tasks:
                task.join()

            res.refresh()
            for i__ in range(len(seeds)):
                ind, ele = res.get()
                next_generation[ind] = ele
            print 'mutate mission', map(lambda x: len(x), missions)


            #inherit
            #calc the fitness
            generation = copy(next_generation)          
            #mp that line
            missions = [[] for i in range(self.cores)]
            for ind, ele in enumerate(generation):
                missions[ind % self.cores].append((ind, ele))

            def f(mission, res):
                for ind, ele in mission:
                    res.put((ind, self.fitness_f(ele)))

            tasks = []
            for ind, ele in enumerate(missions):
                p = mp.Process(target = f, name = 'MP:'+str(ind),
                               args=(ele, res))
                tasks.append(p)
                p.start()

            for task in tasks:
                task.join()
            #the context is copied by other processing
            #restore the context
            res.refresh()
            for ind in range(self.generation_size):
                index, value = res.get()
                fitness[index] = value                    
                if fitness_max == None or self.cmp_f(value, fitness_max):
                    fitness_max = value 
                    best_child = generation[index]
            print 'eval mission', map(lambda x: len(x), missions)


            fitness_sum = sum(fitness)

            end_time = time()
            print 'takes %.5f'% (-start_time + end_time)
            
        self.plot_f(best_child, 'red', False)            
        self.plot_f(self.best_biont, 'blue', True)
        raw_input('Done.')



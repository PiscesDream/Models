from numpy import *

class GA(object):
    
    def __init__(self, fitness_f, terminator, generation_size, generation_init, 
        crossover_rate, crossover_f, mutation_rate, mutation_f, 
        plot = False, plot_f = None, plot_interval = 100,
        cmp_f = None):

        self.fitness_f = fitness_f
        if 'fitness_thresold' in terminator:
            self.fitness_thresold = terminator['fitness_thresold']
            self.iter_maximum = None
        else:
            self.iter_maximum = terminator['iter_maximum']
            self.fitness_thresold = None

        self.generation_size = generation_size
        self.generation_init = generation_init
        self.crossover_rate = crossover_rate
        self.crossover_f = crossover_f
        self.mutation_rate = mutation_rate
        self.mutation_f = mutation_f
        self.plot = plot
        self.plot_f = plot_f
        self.best_fitness = None
        self.plot_interval = plot_interval

        if cmp_f == None:
            self.cmp_f = lambda a,b: a>b
        else:
            self.cmp_f = cmp_f

    def result(self):
        self.best_fitness = None
        self.fit()
        return self.best_biont
    
    def fit(self):        
        generation = [None] * self.generation_size
        fitness = [None] * self.generation_size

        fitness_max = None
        for i in xrange(self.generation_size):
            generation[i] = self.generation_init()
            fitness[i] = self.fitness_f(generation[i])
            if fitness_max == None or self.cmp_f(fitness[i], fitness_max):
                fitness_max = fitness[i]
                best_child = generation[i]
        fitness_sum = sum(fitness)
        
        iteration = 0
        while (self.fitness_thresold == None and iteration < self.iter_maximum) or \
            (self.iter_maximum == None and not self.cmp_f(fitness_max,self.fitness_thresold)):

            if self.best_fitness == None or self.cmp_f(fitness_max, self.best_fitness):
                self.best_fitness = fitness_max
                self.best_biont = best_child

            print '%03dth  generation|\tbest fitness:\t%lf|\tbest child fitness:\t%lf' % (iteration, self.best_fitness,fitness_max)

            if self.plot and iteration % self.plot_interval == 0:
                self.plot_f(best_child, 'red', False)            
                self.plot_f(self.best_biont, 'blue', True)

            iteration += 1
            
            #generation probability
            gen_pr = [None] * self.generation_size
            for i in xrange(self.generation_size):
                gen_pr[i] = fitness[i] * 1.0 / fitness_sum;

            #survival
            next_generation = [None] * self.generation_size
            next_gen_fitness = [None] * self.generation_size
            survival_count = int( (1 - self.crossover_rate) * self.generation_size)
            next_gen_iter = 0
            selecting = sorted(random.rand(survival_count))
            temp_sum = .0
            for index, biont in enumerate(generation):
                temp_sum += gen_pr[index]
                while (next_gen_iter < survival_count and temp_sum >= selecting[next_gen_iter]):
                    next_generation[next_gen_iter] = biont
                    next_gen_iter += 1

            #crossover
            parents_count = (self.generation_size - survival_count) * 2
            parents = [None] * parents_count
            selecting = sorted(random.rand(parents_count))
            parents_iter = 0
            temp_sum = .0
            for index, biont in enumerate(generation):
                temp_sum += gen_pr[index]
                while (parents_iter < parents_count and temp_sum >= selecting[parents_iter]):
                    parents[parents_iter] = biont
                    parents_iter += 1
            parents_iter = 0
            while (next_gen_iter <  self.generation_size):           
                next_generation[next_gen_iter] = self.crossover_f(parents[parents_iter], parents[parents_iter+1])
                parents_iter += 2
                next_gen_iter += 1

            #mutation
            for i in xrange( int( self.generation_size * self.mutation_rate) ):
                mutation_gen = random.randint(0, self.generation_size)
                next_generation[mutation_gen] = self.mutation_f(next_generation[mutation_gen])

            #inherit
            generation = copy(next_generation)
            fitness_max = None
            for i in xrange(self.generation_size):
                fitness[i] = self.fitness_f(generation[i])
                if fitness_max == None or self.cmp_f(fitness[i], fitness_max):
                    fitness_max = fitness[i]
                    best_child = generation[i]
            fitness_sum = sum(fitness)
            
        self.plot_f(best_child, 'red', False)            
        self.plot_f(self.best_biont, 'blue', True)
        raw_input('Done.')

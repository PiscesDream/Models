from numpy import *
from plot import plot

#plot(food, ants, obstacle, pheromone):

class Ant(object):
    
    def __init__(self, position, direction, life, move_speed):
        self.position = position
        self.direction = direction
        self.life = life
        self.passed = []
        self.move_speed = move_speed

    def __getitem__(self, index):
        try:
            return list(self.position)[index]
        except:
            pass

    def move(self):
#easist move
        #self.passed.append(self.position)
        self.position += array(self.direction)*self.move_speed        

        self.life -= 1 
        return self.position
        


class ACO_module(object):
    
    def __init__(self, nests, foods, obstacles,
        fade_rate, 
        gen_speed, move_speed, sense_radius, life_long):

        self.nests = nests
        self.nest_gen = [0] * len(nests)
        self.foods = foods
        self.obstacles = obstacles

        self.fade_rate = fade_rate

        self.gen_speed = gen_speed
        self.move_speed = move_speed
        self.sense_radius = sense_radius
        self.life_long = life_long

        self.pheromone = []
        self.ants = []        

    @staticmethod
    def randomDirection():
        theta = random.rand() * 2*pi
        return [cos(theta), sin(theta)]

    def find_the_food(self):
        
        gen_key = 0
        while True:
            for ind, nest in enumerate(self.nests):
                self.nest_gen[ind] += self.gen_speed[ind]
                while self.nest_gen[ind] > 1:                
                    self.ants.append(Ant(nest, self.randomDirection(), self.life_long[ind], self.move_speed[ind]))
                    self.nest_gen[ind] -= 1
#                    print gen_key

            for ind, ant in enumerate(self.ants): 
                self.pheromone.append(list(ant.move())+[1])
                if ant.life <= 0:
                    self.ants.pop(ind)

            for ind, phe in enumerate(self.pheromone):
                phe[2] -= self.fade_rate
                if phe[2] <= 0:
                    self.pheromone.pop(ind)

            plot(self.foods, self.ants, self.obstacles, self.pheromone)

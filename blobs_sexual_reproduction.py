# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:21:22 2017

@author: ddo003
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:12:40 2017

@author: ddo003
"""


import numpy as np
import pygame 
import matplotlib.pyplot as plt
from threading import Thread
plt.ion()
pygame.init()

#display sizes
display_width = 800
display_height = 600

with open("names.txt", "r") as f:
    name_list = f.readlines()
names = [n.rstrip("\n") for n in name_list]
name_list = None
with open("adjectives.txt", "r") as f:
    ad_list = f.readlines()
ads = [n.rstrip("\n").upper() for n in ad_list]
ad_list = None


#Some colors
blue = (0,0,255)
red = (255,0,0)
green = (0,255,0)
black = (0,0,0)
white = (255,255,255)

#some game variables and functions:
max_velocity = 2
max_force = 0.5 

def magnitude_calc(vector):
    x = 0
    for i in vector:
        x += i**2
    magnitude = x**0.5
    return(magnitude)
 
def normalize(vector):
    magnitude = magnitude_calc(vector)
    if magnitude != 0:
        vector = vector/magnitude
    return(vector)

    

class new_blob:
    def __init__(self, dna = None):
        self.dna = dna
        self.size = 5
        self.health = 100
        self.age = 1
        self.max_force = 0.5
        self.max_velocity = 2
        self.dead = False
        self.mating = False
        self.mate_selected = False
        self.fitness = self.age + self.health*3
        self.name = np.random.choice(ads)+" "+np.random.choice(names)
        
        self.color = (np.max(int((255-(self.health * 2.55))),0),0,int((self.health * 2.55)))
        self.velocity = np.array([np.random.uniform(-max_velocity, max_velocity),
                                  np.random.uniform(-max_velocity, max_velocity)], dtype='float64')
       
        self.acceleration = np.array([0, 0], dtype='float64')
        
        
        if self.dna == None:
            self.vision = np.random.uniform(1,50)        
            self.chance = np.random.uniform(0.0001, 0.01)
            self.attraction_to_food = np.random.random()
            self.attraction_to_poison = np.random.random()
            self.pos = np.array([np.random.uniform(0,display_width),
                                 np.random.uniform(0,display_height)],dtype='float64')
            self.mutation_rate = np.random.uniform(0.1, 0.20)
            self.max_rand_dist = np.random.uniform(10, 800)
            self.dna = [self.vision, self.chance, self.attraction_to_food, self.mutation_rate,self.pos, self.max_rand_dist, self.attraction_to_poison]
        else:
            self.vision = self.dna[0]
            self.chance = self.dna[1]
            self.attraction_to_food = self.dna[2]
            self.mutation_rate = self.dna[3]
            self.pos = self.dna[4]
            self.max_rand_dist = self.dna[5]
            self.attraction_to_poison = self.dna[6]
        
        self.random_pos = np.array([np.random.uniform(self.pos[0]-self.max_rand_dist, self.pos[0]+self.max_rand_dist), np.random.uniform(self.pos[1]-self.max_rand_dist,self.pos[1]+self.max_rand_dist)])
        if self.random_pos[0] < 0:
            self.random_pos[0] = 0
        if self.random_pos[0] > display_width:
            self.random_pos[0] = display_width
        if self.random_pos[1] < 0:
            self.random_pos[1] = 0
        if self.random_pos[1] > display_height:
            self.random_pos[1] = display_height
        
    def seek(self, target):
        desired_vel = np.add(target, -self.pos)
        desired_vel = normalize(desired_vel)*self.max_velocity
        steering_force = np.add(desired_vel, -self.velocity)
        steering_force = normalize(steering_force)*self.max_force
        
        return(steering_force)
    
    def select_mate(self):
        blob_pos = [x.pos for x in blobs if x.pos.tolist() != self.pos.tolist()]
        dists = np.array([np.linalg.norm(x-self.pos) for x in blob_pos])
        closest_mate = blobs[np.argmin(dists)]
        self.mate_selected = closest_mate
        if self.name == closest_mate.name:
                print(self.name+" tried asexual reproduction and failed")
                closest_mate = blobs[np.argmax(dists)]
        pygame.draw.line(gameDisplay, (255,255,0), (int(self.pos[0]), int(self.pos[1])), (int(closest_mate.pos[0]), int(closest_mate.pos[1])),1)
        return closest_mate.pos.tolist()

            
    def select_target(self, food, poison):
        dists = np.array([np.linalg.norm(x-self.pos) for x in food])
        if np.min(dists) < self.size:
                self.eat_food(np.argmin(dists))
                return np.array([np.random.randint(0,display_width), np.random.randint(0,display_height)])[0]
            
        if len(poison) > 0:
            dists_poison = np.array([np.linalg.norm(x-self.pos) for x in poison])
                               
            if np.min(dists_poison) < self.size:
                self.eat_poison(np.argmin(dists_poison))
                return np.array([np.random.randint(0,display_width), np.random.randint(0,display_height)])[0]
            
       
            if np.min(dists_poison) < self.vision and np.random.random() < self.attraction_to_poison:
                target = poison[np.argmin(dists_poison)]
                return target
            elif np.min(dists) < self.vision and np.random.random() < self.attraction_to_food:
                target = food[np.argmin(dists)]
                return target
            else:
                return False
        else:
            if np.min(dists) < self.vision and np.random.random() < self.attraction_to_food:
                target = food[np.argmin(dists)]
                return target
            else:
                return False
    def new_random_pos(self):
        self.random_pos = np.array([np.random.uniform(self.pos[0]-self.max_rand_dist, self.pos[0]+self.max_rand_dist), np.random.uniform(self.pos[1]-self.max_rand_dist,self.pos[1]+self.max_rand_dist)])
        if self.random_pos[0] < 0:
            self.random_pos[0] = 0
        if self.random_pos[0] > display_width:
            self.random_pos[0] = display_width
        if self.random_pos[1] < 0:
            self.random_pos[1] = 0
        if self.random_pos[1] > display_height:
            self.random_pos[1] = display_height
        return(self.random_pos)
    
    def scavenge(self):
        if np.linalg.norm(self.random_pos - self.pos) <= self.size:
            self.new_random_pos()
        if np.random.random() < self.chance:
            self.new_random_pos()
        self.acceleration += self.seek(self.random_pos)
        self.velocity += self.acceleration
        self.velocity = normalize(self.velocity)*max_velocity
        self.pos += self.velocity 
    
    def set_color(self):
        if self.health >= 100:
            c = (0,0,255)
        elif self.health <= 0:
            c = (255,0,0)
        else:
            c = ( np.max([(255-(int(self.health*2.55))),0]),
                  0,
                  np.min([255, int(self.health*2.55)])
                  )
        self.color = c

    def update(self):
        self.age += 1
        self.set_color()
        if self.age % 50 == 0:
            self.health -= 5
        if self.health <= 0:
            self.die()
        self.fitness = self.health*3+self.age
        
        if self.mating == True:
            self.max_velocity = 2
            if self.mate_selected == False:
                target = self.select_mate()

            else:
                if self.mate_selected.dead != True:
                    target = self.mate_selected.pos
                else:
                    target = self.select_mate()
            pygame.draw.line(gameDisplay, (255,255,0), (int(self.pos[0]), int(self.pos[1])), (int(target[0]), int(target[1])),1)

            self.acceleration += self.seek(target)
            self.velocity += self.acceleration
            self.velocity = normalize(self.velocity)*max_velocity
            self.pos += self.velocity
            
            if np.linalg.norm(target - self.pos) < self.size:
                self.procreate(self.mate_selected)
            
        else:
            self.max_velocity = 2.0
            target = self.select_target(food, poison)

        
        
            if target == False:
                self.scavenge()
            else:
                self.acceleration += self.seek(target)
                self.velocity += self.acceleration
                self.velocity = normalize(self.velocity)*max_velocity
                self.pos += self.velocity
        self.target = target
        
    def eat_food(self, eaten_food):
        food.pop(eaten_food)
        self.health += 20
        
    def eat_poison(self, eaten_poison):
        poison.pop(eaten_poison)
        self.health *= 0.5
        self.health -= 5
        
    def die(self):
        self.dead = True
        food.append([int(self.pos[0]), int(self.pos[1])])
        print(self.name + " died at the age of ", self.age // 60, " seconds.")

    def meiosis(self):
        offspring_dna = []
        if np.random.random() < self.mutation_rate:
            offspring_vision = np.random.uniform(1,50)
        else:
            offspring_vision = self.vision
        if np.random.random() < self.mutation_rate:
            offspring_chance = np.random.uniform(0.0001, 0.01)
        else:
            offspring_chance = self.chance
        if np.random.random() < self.mutation_rate:  
            offspring_attraction_to_food = (np.random.random())
        else:
            offspring_attraction_to_food = self.attraction_to_food
        if np.random.random() < self.mutation_rate:
            offspring_mutation_rate = np.random.uniform(0.1,0.2)
        else:
            offspring_mutation_rate = self.mutation_rate
        if np.random.random() < self.mutation_rate:
            offspring_max_rand_dist = np.random.uniform(10, 800)
        else:
            offspring_max_rand_dist = self.max_rand_dist
        if np.random.random() < mutation_rate:
            offspring_attraction_to_poison = np.random.random()
        else:
            offspring_attraction_to_poison = self.attraction_to_poison
        offspring_pos = self.pos + + np.array([np.random.uniform(-10,10),np.random.uniform(-10,10)])
        
        offspring_dna = [offspring_vision, 
                         offspring_chance, 
                         offspring_attraction_to_food, 
                         offspring_mutation_rate, 
                         offspring_pos, 
                         offspring_max_rand_dist, 
                         offspring_attraction_to_poison]
        return offspring_dna
    
    def procreate(self, mate):
        if self.name == mate.name:
            self.mating = False
            print(self.name+" tried asexual reproduction and failed")
        else:
        
            gamete = self.meiosis()
            mate_gamete = mate.meiosis()
            offspring_dna = []
            for ii in range(len(gamete)):
                choice = np.random.choice([0,1])
                if choice == 0:
                    offspring_dna.append(gamete[ii])
                else:
                    offspring_dna.append(mate_gamete[ii])
            baby = new_blob(dna = offspring_dna)
            blobs.append(baby)
            self.mating = False
            print(self.name + " and " + mate.name + " had a baby named " + baby.name)
        
        
        
    def draw(self):
        pygame.draw.circle(gameDisplay, self.color, (int(self.pos[0]),int(self.pos[1])), int(self.size))
        pygame.draw.circle(gameDisplay, green, (int(self.pos[0]),int(self.pos[1])), int(self.vision), 1)
        pygame.draw.line(gameDisplay, green, (int(self.pos[0]), int(self.pos[1])), (int(self.pos[0] + (self.velocity[0]*self.dna[2]*10)), int(self.pos[1] + (self.velocity[1]*self.dna[2]*10))), 1)
        pygame.draw.line(gameDisplay, red, (int(self.pos[0]), int(self.pos[1])), (int(self.pos[0] + (self.velocity[0]*self.dna[6]*10)), int(self.pos[1] + (self.velocity[1]*self.dna[6]*10))), 1)
     
      
            

food = [[np.random.randint(0,display_width), np.random.randint(0,display_height)] for _ in range(20)]

poison = [[np.random.randint(0,display_width), np.random.randint(0,display_height)] for _ in range(5)]

food_rate = 0.04
poison_rate = 0.02
max_poison = 15
spawning_rate = 0.003
random_spawning_rate = 0.0001
max_population = 50
initial_population = 10
mutation_rate = 0.05


blobs = [new_blob() for _ in range(initial_population)]
for blob in blobs:
    print(blob.dna)




#initialize some pygame stuff
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Look, a window!")
clock = pygame.time.Clock()
running = True
population_fitness = []
population_mean_age = []
population_size = []

def plot_fitness():
    global population_fitness, running
    plt.figure()
    while running:
        if len(population_fitness) > 120:    
            while running:
                plt.plot(population_fitness[-120:])
                plt.pause(0.01)
                plt.clf()
    

while running:
    gameDisplay.fill((0,0,0))
    
    for f in food:
        pygame.draw.circle(gameDisplay, green, (f[0],f[1]), 3)
    for p in poison:
        pygame.draw.circle(gameDisplay, red, (p[0],p[1]), 3)
    if np.random.random() < food_rate:
        food.append([np.random.randint(0,display_width), np.random.randint(0,display_height)])
    if np.random.random() < poison_rate:
        poison.append([np.random.randint(0,display_width), np.random.randint(0,display_height)])
    if len(poison) > 15:
        poison = poison[len(poison)-15:]
    #Give us the option of closing out:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    cur_pop_fitness = 0
    cur_mean_age = 0
    for blob in blobs:
        if blob.dead:
            blobs.remove(blob)
        blob.draw()
        blob.update()       
        if blob.dead:
            blobs.remove(blob)
        cur_pop_fitness += blob.fitness
        cur_mean_age += blob.age
    
    population_fitness.append(cur_pop_fitness)
    population_mean_age.append(cur_mean_age/len(blobs))
    population_size.append(len(blobs))

    
    if len(blobs) < max_population and len(blobs) > 1:
        if np.random.random() < spawning_rate:
            fitnesses = np.array([b.fitness for b in blobs])
            fittest_blobs = np.where(fitnesses>(fitnesses.mean()))
            if len(blobs) > 3:
                lucky_blob = blobs[np.random.choice(fittest_blobs[0])]
            else:
                lucky_blob = np.random.choice(blobs)
            lucky_blob.mating = True
    if np.random.random() < random_spawning_rate:
        immigrant = new_blob()
        blobs.append(immigrant)
        print("One of them immigrants named "+immigrant.name+" just arrived to steal our jobs!")
    pygame.display.update()
    clock.tick(60)

pygame.quit()


def plot_summary():
    param_names = ["fitness", "mean age", "population_size"]
    parameters = [population_fitness, population_mean_age, population_size]
    xs = [i/60 for i in range(len(parameters[0]))]
    c = 1
    plt.figure()
    for p in parameters:
        plt.subplot(3,1,c)
        plt.title(param_names[c-1])
        plt.xlabel("Time")
        plt.ylabel(param_names[c-1])
        plt.plot(xs, p)
        c+=1
    plt.tight_layout()

plot_summary()
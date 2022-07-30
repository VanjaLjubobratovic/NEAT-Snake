from distutils.command.config import config
from genericpath import isfile
from glob import glob
from importlib.resources import path
import os
import random
import numpy as np
from collections import deque
from snakeGame import SnakeGameAI, Direction, Point
from plotter import plot
import sys 
import pickle
import math
import neat
from neat import nn, population
import threading

BLOCK_SIZE = 10
MAX_GENERATIONS = 30

NEAR_FOOD_REWARD = 0.25
LOOP_PUNISHMENT = -0.05

generation_number = 0
plot_mean_scores = []
plot_best_scores = []
plot_generation_fitness = []
plot_mean_generation_fitness = []

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_inputs(game):
        #snake will look for danger BLOCK_SIZE in front of itself
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger in front
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #danger left
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #move direction, only one is true
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food location
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y #food down
        ]

        return np.array(state, dtype=int) #array with booleans converted to 0 or 1

def save_best_generation_instance(instance, filename='trained/best_generation_instances.pickle'):
        instances = []
        if os.path.isfile(filename):
            instances = load_object(filename)
        instances.append(instance)
        save_object(instances, filename)

def eval_fitness(genomes, config):
    global best_fitness
    global generation_number
    global pop

    best_instance = None
    genome_number = 0
    best_fitness = -30
    total_score = 0

    for _, g in genomes:
        game = SnakeGameAI()
        net = nn.FeedForwardNetwork.create(g, config)
        score = 0.0
        additional_points = 0.0

        while True:
            action = [0, 0, 0]
            inputs = get_inputs(game)
            output = net.activate(inputs)
            action[np.argmax(output)] = 1

            # print("INPUTS: ", inputs)
            # print("OUTPUTS: ", output)
            # print("ACTION: ", action)
            # print("-----------------------------")

            reward, game_over, score = game.play_step(action)

            head = game.head
            food = game.food
            distance_to_food = np.sqrt(np.square(head.x - food.x) + np.square(head.y - food.y)) / BLOCK_SIZE
            
            # Rewarding snake for getting close to food
            if distance_to_food <= 1:
                additional_points += NEAR_FOOD_REWARD
            
            # Punishing snake for spinning in place
            if head in game.past_points:
                additional_points += LOOP_PUNISHMENT

            if game_over or additional_points < -1.5:
                break

        g.fitness = round(score * 3 + additional_points, 2)

        if not best_instance or g.fitness > best_fitness:
            best_instance = {
            'num_generation': generation_number,
            'fitness': g.fitness,
            'score' : score,
            'genome': g,
            'net': net,
        }

        best_fitness = max(best_fitness, g.fitness)
        print(f"Generation {generation_number} \tGenome {genome_number} \tFitness {g.fitness} \tBest fitness {best_fitness} \tScore {score}")
        genome_number += 1
        total_score += score

        plot_generation_fitness.append(g.fitness)
        total_generation_fitness = np.sum(plot_generation_fitness, 0)
        mean_generation_fitness = round(total_generation_fitness / len(plot_generation_fitness), 2)
        plot_mean_generation_fitness.append(mean_generation_fitness)

        plot(plot_generation_fitness, plot_mean_generation_fitness)


    #save_best_generation_instance(best_instance)
    generation_number += 1

    # if generation_number % 10 == 0:
    #     save_object(pop, 'trained/population.dat')
    #     print("Exporting population")
    
    #plotting
    plot_best_scores.append(best_instance.get('score'))
    #total_score = np.sum(plot_best_scores, 0)
    mean_score = total_score / genome_number
    plot_mean_scores.append(mean_score)
    #print("MEAN: ", plot_mean_scores)
    #plot(plot_best_scores, plot_mean_scores)


def train():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-neat.txt")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    pop = population.Population(config)
    pop.run(eval_fitness, 100)

if __name__ == '__main__':
    train()





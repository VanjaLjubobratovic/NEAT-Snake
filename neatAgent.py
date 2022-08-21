
from genericpath import isfile
from glob import glob
from importlib.resources import path
import os
from time import sleep
import numpy as np
from collections import deque
from snakeGame import SnakeGameAI, Direction, Point
from plotter import plot
import pickle
import neat
from neat import nn, population

BLOCK_SIZE = 10
MAX_GENERATIONS = 30

NEAR_FOOD_REWARD = 0.15
LOOP_PUNISHMENT = -0.5
FOOD_REWARD_MULTIPLIER = 10
LIVING_PUNISHMENT = -0.2

best_instance_list = deque(maxlen=1)
best_instance_list.append(None)

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
        food = game.food

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            game.is_collision(point_r),
            game.is_collision(point_l),
            game.is_collision(point_u),
            game.is_collision(point_d),

            # KOMENTAR
            # Ako ovo otkomentirate, morate u config-neat.txt promijeniti num_inputs na 12.
            # Move direction, only one is true
            # dir_l,
            # dir_r,
            # dir_u,
            # dir_d,

            #Food location
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y #food down
        ]

        return np.array(state, dtype=float) #array with booleans converted to 0 or 1

def save_best_generation_instance(instance, file_name='best_instance_neat.pickle'):
    net_folder_path = "./neural-net"
    if not os.path.exists(net_folder_path):
        os.makedirs(net_folder_path)
    file_name = os.path.join(net_folder_path, file_name)
    save_object(instance, file_name)

def eval_fitness(genomes, config):
    global best_fitness
    global generation_number
    global pop

    best_instance = None
    best_fitness = -1000
    total_score = 0

    plot_generation_fitness = []

    for _, g in genomes:
        score = 0.0
        additional_points = 0.0

        # KOMENTAR
        # Evaluaciju svake jedinke se ponovi 10 puta kako bi dobili bolju procjenu koliko je dobra ta jedinka.
        for _ in range(10):
            game = SnakeGameAI(False, 10_000)
            net = nn.FeedForwardNetwork.create(g, config)

            while True:
                # KOMENTAR
                # Agent ima na raspolaganju 4 akcije: gore, dolje, lijevo, desno
                # Akcije nisu relativne zmiji već globalnom koordinatnom sustavu

                action = [0, 0, 0, 0]
                inputs = get_inputs(game)
                output = net.activate(inputs)
                action[np.argmax(output)] = 1
                
                head = game.head
                food = game.food
                distance_to_food_prev = np.sqrt(np.square(head.x - food.x) + np.square(head.y - food.y)) / BLOCK_SIZE

                reward, game_over, delta_score = game.play_step(action)

                head = game.head
                food = game.food
                distance_to_food_current = np.sqrt(np.square(head.x - food.x) + np.square(head.y - food.y)) / BLOCK_SIZE

                # KOMENTAR
                # Agent dobiva pozitivne bodove kada se približava hrani, a dvostruko brze dobiva negativne bodove kada se udaljava.
                # Kada sakupi hranu, hrana ce se teleportirati negdje drugdje na ploci i taj slucaj ignoriramo sa uvjetom (reward != 10) jer
                # u tom slucaju nema smisla nagradivati/kaznjavati agenta.
                if reward != 10:
                    delta = distance_to_food_prev - distance_to_food_current
                    if delta > 0:
                        additional_points += delta
                    else:
                        additional_points += delta * 2

                if game_over:
                    break
            score += delta_score

        # KOMENTAR
        # Izracunavamo prosjecan fitness i score.
        additional_points = additional_points / 10
        score = score / 10

        #additional_points = additional_points / 10 + score * FOOD_REWARD_MULTIPLIER

        # KOMENTAR
        # Fitness ovisi samo o additional_points.
        g.fitness = round(additional_points, 2)

        if not best_instance or g.fitness > best_fitness:
            best_instance = {
            'num_generation': generation_number,
            'fitness': g.fitness,
            'score' : score,
            'genome': g,
            'net': net,
            }
            best_instance_list.append(best_instance)
            

        best_fitness = max(best_fitness, g.fitness)
        total_score += score

        plot_generation_fitness.append(g.fitness)
    
    total_generation_fitness = np.sum(plot_generation_fitness, 0)
    mean_generation_fitness = round(total_generation_fitness / len(plot_generation_fitness), 2)
    plot_mean_generation_fitness.append(mean_generation_fitness)

    plot_best_scores.append(best_instance.get('score'))

    # Mozda bi se ovo moglo malo ljepse napravit
    # Prvi argument je lista tuplova u obliku (lista_mjerenja, label)
    plot([(plot_best_scores, "Best gen. score"), (plot_mean_generation_fitness, "mean gen. fitness")], "generations", "score", -50, "neat_scores.png")

    generation_number += 1

def test_trained_net():
    scores = []
    # KOMENTAR
    # Izvodi se igra 10 puta i ispisuje se prosjecan score.        
    for _ in range(10):
        sleep(2)
        game = SnakeGameAI(True, 100)
        net = best_instance_list[0].get('net')

        while True:
            action = [0, 0, 0, 0]
            inputs = get_inputs(game)
            output = net.activate(inputs)
            action[np.argmax(output)] = 1

            reward, game_over, score = game.play_step(action)
            if game_over:
                scores.append(score)
                break
    avg_score = sum(scores) / len(scores)
    print(scores)
    print("Average score: {}".format(avg_score))


def train():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-neat.txt")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    pop = population.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(100))

    pop.run(eval_fitness, 1000)


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(e)
    finally:
        print("Training done")
        if(best_instance_list):
            save_best_generation_instance(best_instance_list[0].get('net'))
            test_trained_net()
        else:
            print("No best instance saved! Exiting...")






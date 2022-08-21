import pickle
from time import sleep
import sys
import neat
import torch
import torch.nn, torch.optim
import neat.nn, neat.population
from neatAgent import get_inputs
import numpy as np
from model import Linear_QNet, QTrainer

from snakeGame import Point, SnakeGameAI, Direction

config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            "config-neat.txt")

BLOCK_SIZE = 10

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


def play_game():
    path = "./neural-net/" + str(sys.argv[1])
    neat = "neat" in path

    scores = []
    for game_num in range(10):
        sleep(2)
        game = SnakeGameAI(True, 100)

        if neat:
            net = load_object(path).get('net')
        else:
            net = Linear_QNet(8, 256, 4)
            net.load_state_dict(torch.load(path))

        while True:
            action = [0, 0, 0, 0]
            inputs = get_inputs(game)

            if neat:
                output = net.activate(inputs)
            else:
                state0 = torch.tensor(inputs, dtype=torch.float)
                output = net(state0)
                output = output.detach().numpy()
                
            action[np.argmax(output)] = 1

            reward, game_over, score = game.play_step(action)
            if game_over:
                scores.append(score)
                break

    avg_score = sum(scores) / len(scores)
    print(scores)
    print("Average score: {}".format(avg_score))
        


if __name__ == "__main__":
    play_game()
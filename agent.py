from re import M
from tkinter import Y
import torch
import random
import numpy as np
from collections import deque
from snakeGame import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate
BLOCK_SIZE = 10

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 #discount
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(8, 256, 4) #state vector, hidden layer, output vector (action)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, game):
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

        # Translating food coordinates to snake coordinate system
        food_tr = Point(game.food.x - head.x, game.food.y - head.y)


        state = [
            game.is_collision(point_r),
            game.is_collision(point_l),
            game.is_collision(point_u),
            game.is_collision(point_d),

            # KOMENTAR
            # Ako ovo otkomentirate, morate u config-neat.txt promijeniti num_inputs na 12.
            # Move direction, only one is true
            #dir_l,
            #dir_r,
            #dir_u,
            #dir_d,

            #Food location
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y #food down

            # abs(food_tr.x) > abs(food_tr.y),
            # abs(food_tr.x) < abs(food_tr.y)


            # game.food.x > game.head.x and game.food.y < game.head.y, #Quadrant I
            # game.food.x < game.head.x and game.food.y < game.head.y, #Quadrant II
            # game.food.x < game.head.x and game.food.y > game.head.y, #Quadrant III
            # game.food.x > game.head.x and game.food.y > game.head.y #Quadrant IV


            # game.food.x > game.head.x and game.food.y < game.head.y and abs(food_tr.x) > abs(food_tr.y), #Octant I
            # game.food.x > game.head.x and game.food.y < game.head.y and abs(food_tr.x) < abs(food_tr.y), #Octant II

            # game.food.x < game.head.x and game.food.y < game.head.y and abs(food_tr.x) < abs(food_tr.y), #Octant III
            # game.food.x < game.head.x and game.food.y < game.head.y and abs(food_tr.x) > abs(food_tr.y), #Octant IV

            # game.food.x < game.head.x and game.food.y > game.head.y and abs(food_tr.x) > abs(food_tr.y), #Octant V
            # game.food.x < game.head.x and game.food.y > game.head.y and abs(food_tr.x) < abs(food_tr.y), #Octant VI

            # game.food.x > game.head.x and game.food.y > game.head.y and abs(food_tr.x) < abs(food_tr.y), #Octant VII
            # game.food.x > game.head.x and game.food.y > game.head.y and abs(food_tr.x) > abs(food_tr.y) #Octant VIII

        ]

        return np.array(state, dtype=float) #array with booleans converted to 0 or 1


    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) #popleft if MAX_MEMORY is reached automatically

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        #random moves: explore / exploit
        self.epsilon = 80 - self.num_games #hardcoded
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            #make a proper move
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(False, 10_000)

    while True:
        #get old state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        #member
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            #train long memory
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print("Game ", agent.num_games, "Score ", score, "Record ", record)

            #Plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
if __name__ == '__main__':
    train()
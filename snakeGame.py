from collections import namedtuple
from enum import Enum
from random import random
import numpy
import pygame
import time
import random
from collections import deque
import numpy as np


pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

#drawing colors
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
white = (255, 255, 255)
black = (0, 0, 0)
yellow = (255, 255, 102)

snake_color = yellow

snake_block_dimens = 10
#snake_speed = 10000

MAX_STEPS_CONST = 200
MAX_STEP_MEMORY = 5
near_food_score = 0.2
loop_punishment = 0.05

#screen dimens
dis_w = 800
dis_h = 600

font_style = pygame.font.SysFont(None, 30)

Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    def __init__(self, draw, speed):
        self.draw = draw
        self.snake_speed = speed
        if self.draw:
            self.dis = pygame.display.set_mode((dis_w, dis_h))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Smort Snek game")
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(dis_w / 2, dis_h / 2)
        self.snake = [self.head,
                        Point(self.head.x - snake_block_dimens, self.head.y),
                        Point(self.head.x - 2 * snake_block_dimens, self.head.y)]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        self.time_spent = 0.0
        self.past_points = deque(maxlen=MAX_STEP_MEMORY)
    
    def place_food(self):
        #food block coords
        foodx = round(random.randrange(0, dis_w - snake_block_dimens) / snake_block_dimens) * snake_block_dimens
        foody = round(random.randrange(0, dis_h - snake_block_dimens) / snake_block_dimens) * snake_block_dimens

        self.food = Point(foodx, foody)

        if(self.food in self.snake):
            self.place_food()

    #shows message on top of the screen
    def show_message(self, msg, color = red, msg_x = 0, msg_y = 0):
        mess = font_style.render(msg, True, color)
        self.dis.blit(mess, [msg_x, msg_y])
        pygame.display.update()

    def move(self, action):

        # [up, down, right, left] snake action
        # KOMENTAR
        # Ovo sam prilagodio

        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = Direction.UP
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_dir = Direction.DOWN
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = Direction.RIGHT
        elif np.array_equal(action, [0, 0, 0, 1]):
            new_dir = Direction.LEFT
    
        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += snake_block_dimens
        elif self.direction == Direction.LEFT:
            x -= snake_block_dimens
        elif self.direction == Direction.UP:
            y -= snake_block_dimens
        elif self.direction == Direction.DOWN:
            y += snake_block_dimens
        
        self.head = Point(x, y)

    def update_ui(self):
        self.dis.fill(black)

        #drawing each snake block
        for block in self.snake:
            pygame.draw.rect(self.dis, snake_color, pygame.Rect(block.x, block.y, snake_block_dimens, snake_block_dimens))
        
        pygame.draw.rect(self.dis, red, pygame.Rect(self.food.x, self.food.y, snake_block_dimens, snake_block_dimens))
        self.show_message("Score: " + str(self.score))

    def is_collision(self, pt = None):
        if pt is None:
            pt = self.head
        
        #wall
        if pt.x > dis_w - snake_block_dimens or pt.x < 0 or pt.y > dis_h - snake_block_dimens or pt.y < 0:
            return True
        #noms tail
        if pt in self.snake[1:]:
            return True
        
        return False

    def play_step(self, action):
        self.frame_iteration += 1

        #user input check
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.past_points.append(self.head)
        
        #Move step
        self.move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > MAX_STEPS_CONST:
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        #Place new food or move
        if self.head == self.food:
            self.frame_iteration = 0
            self.score += 1
            reward = 10
            self.place_food()
        else:
            #remove last tail block after moving head
            self.snake.pop()

        # Adding points for getting close to food
        # distance_to_food = np.sqrt(np.square(self.head.x - self.food.x) + np.square(self.head.y - self.food.y))
        # if distance_to_food <= 0:
        #     self.score += near_food_score

        # Punishing the snake for spinning in place
        # if self.head in self.past_points:
        #     self.score -= loop_punishment

        self.score = round(self.score, 2)

        #Update UI and clock
        if self.draw:
            self.update_ui()
        self.clock.tick(self.snake_speed)
        return reward, game_over, self.score
from random import random
import pygame
import time
import random

pygame.init()

#drawing colors
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
white = (255, 255, 255)
black = (0, 0, 0)
yellow = (255, 255, 102)

snake_color = yellow

#screen dimens
dis_w = 800
dis_h = 600

snake_block_dimens = 10
snake_speed = 20

clock = pygame.time.Clock()

dis = pygame.display.set_mode((dis_w, dis_h))
pygame.display.set_caption("Smort Snek game")

font_style = pygame.font.SysFont(None, 30)

#shows message on top of the screen
def message(msg, color, msg_x = 0, msg_y = 0):
    mess = font_style.render(msg, True, color)
    dis.blit(mess, [msg_x, msg_y])
    pygame.display.update()

def snake(snake_block_dimens, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, snake_color, [x[0], x[1], snake_block_dimens, snake_block_dimens])

def game_loop():
    game_over = False
    game_close = False

    x1_change = 0
    y1_change = 0

    #snake head coordinates
    x1 = dis_w / 2
    y1 = dis_h / 2

    #food block coords
    foodx = round(random.randrange(0, dis_w - snake_block_dimens) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_h - snake_block_dimens) / 10.0) * 10.0

    snake_list = []
    snake_len = 1

    score = 0
    time_spent = 0.0

    while not game_over:
        while game_close == True:
            dis.fill(black)
            message("Game over! Q-quit or C-play again", red)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_close = False
                        game_over = True
                    elif event.key == pygame.K_c:
                        game_loop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and x1_change <= 0:
                    x1_change = -snake_block_dimens
                    y1_change = 0
                elif event.key == pygame.K_RIGHT and x1_change >= 0:
                    x1_change = snake_block_dimens
                    y1_change = 0
                elif event.key == pygame.K_DOWN and y1_change >= 0:
                    x1_change = 0
                    y1_change = snake_block_dimens
                elif event.key == pygame.K_UP and y1_change <= 0:
                    x1_change = 0
                    y1_change = -snake_block_dimens

        #wall hit detection
        if x1 >= dis_w or x1 < 0 or y1 >= dis_h or y1 < 0:
            game_close = True

        x1 += x1_change
        y1 += y1_change
        dis.fill(black)
        
        #screen, color, [x, y, block_w, block_w]
        pygame.draw.rect(dis, blue, [foodx, foody, snake_block_dimens, snake_block_dimens])

        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)

        #deletes unnecessary block
        if len(snake_list) > snake_len:
            del snake_list[0]
        
        #ate itself check
        for x in snake_list[:-1]:
            if x == snake_head:
                game_close = True
        
        snake(snake_block_dimens, snake_list)

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            print("NOM!")
            foodx = round(random.randrange(0, dis_w - snake_block_dimens) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_h - snake_block_dimens) / 10.0) * 10.0
            snake_len += 1
            score += 1
        
        message("Your score: " + str(score), red)
        message("Your time: " + str(round(time_spent, 2)) + "s", red, 0, 35)

        time_spent += snake_speed / 1000.0
        clock.tick(snake_speed)

    pygame.quit()
    quit()

game_loop()
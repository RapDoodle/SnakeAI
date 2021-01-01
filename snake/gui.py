import pygame
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from settings import settings

import common.constant as const

class SnakeGameGUI():

    def __init__(self, game = None):

        # Register the controller
        self.game = game

        # Color scheme
        self.background_color = const.BLACK

        self.window_width = settings['NUM_COLS'] * settings['GRID_SIZE'] + (settings['NUM_COLS'] - 1) * settings['GRID_MARGIN'] + settings['BODY_MARGIN'] * 2 + settings['BODY_BORDER_WIDTH'] * 2
        self.window_height = self.window_width

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.running = True

        self.length = 1

        pygame.display.set_caption('Snake AI')

    def paint_board(self):
        x1 = 0
        y1 = 0
        x2 = self.window_width
        y2 = self.window_height
        pygame.draw.rect(self.screen, self.background_color,[x1 , y1, x2, y2], 0)

        x1 = settings['BODY_MARGIN'] - settings['BODY_BORDER_WIDTH']
        y1 = x1
        x2 = self.window_width - 2*(settings['BODY_MARGIN'] - settings['BODY_BORDER_WIDTH'])
        y2 = x2
        pygame.draw.rect(self.screen, const.WHITE, [x1, y1, x2, y2], settings['BODY_BORDER_WIDTH'])

    def draw_block(self, r, c, color):
        x1 = settings['BODY_MARGIN'] + c*settings['GRID_SIZE'] + (c+1)*settings['GRID_MARGIN']
        y1 = settings['BODY_MARGIN'] + r*settings['GRID_SIZE'] + (r+1)*settings['GRID_MARGIN']
        pygame.draw.rect(self.screen, color, pygame.Rect([x1, y1, settings['GRID_SIZE'], settings['GRID_SIZE']]))

    def load(self):
        root = tk.Tk()
        root.withdraw()

        replay_path = filedialog.askopenfilename()

        if len(replay_path) == 0:
            return

        log = pd.read_csv(replay_path)

        row = log.iloc[0, :]
        init_pos = (row['pos_r'], row['pos_c'])
        init_dir = (row['pos_dr'], row['pos_dc'])
        inverse_direction = (-1*init_dir[0], -1*init_dir[1])
        new_r = init_pos[0]
        new_c = init_pos[1]
        self.game.positions.append((new_r, new_c))
        self.game.game_init(replay_mode = True, init_len = 3)
        for _ in range(3-1):
            new_r = new_r + inverse_direction[0]
            new_c = new_c + inverse_direction[1]
            self.game.positions.append((new_r, new_c))
        self.game.turn((row['pos_dr'], row['pos_dc']))
        self.game.move(replay_mode = True)
        self.paint_board()

        for _, row in log.iterrows():
            self.paint_board()
            
            if row['type'] == const.MOVE or row['type'] == const.EAT:
                self.game.turn((row['pos_dr'], row['pos_dc']))
                self.game.move(replay_mode = True)
                pygame.time.wait(10)
                
            elif row['type'] == const.RAND_APPLE:
                self.game.set_apple(row['pos_r'], row['pos_c'])

            elif row['type'] == const.RAND_BANANA:
                self.game.set_banana(row['pos_r'], row['pos_c'])

            elif row['type'] == const.SPAWN:
                pass

            elif row['type'] == const.END:
                self.game.end = True
                break

            for pos in self.game.apples:
                self.draw_block(pos[0], pos[1], const.RED)

            for pos in self.game.bananas:
                self.draw_block(pos[0], pos[1], const.YELLOW)
                
            for pos in self.game.positions:
                self.draw_block(pos[0], pos[1], const.WHITE)

            pygame.display.update()
            pygame.event.pump()

    def spin(self):
        clock = pygame.time.Clock()

        self.paint_board()

        while self.running:
            clock.tick(10)
            pygame.event.pump()
            self.paint_board()
            for pos in self.game.apples:
                self.draw_block(pos[0], pos[1], const.RED)
            for pos in self.game.bananas:
                self.draw_block(pos[0], pos[1], const.YELLOW)
            for pos in self.game.positions:
                self.draw_block(pos[0], pos[1], const.WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if self.game is not None:
                        if event.key == pygame.K_UP:
                            self.game.turn(const.UP)
                        elif event.key == pygame.K_DOWN:
                            self.game.turn(const.DOWN)
                        elif event.key == pygame.K_LEFT:
                            self.game.turn(const.LEFT)
                        elif event.key == pygame.K_RIGHT:
                            self.game.turn(const.RIGHT)
                        elif event.key == pygame.K_r:
                            self.game.game_init()
                        elif event.key == pygame.K_s:
                            self.game.save()
                        elif event.key == pygame.K_l:
                            self.load()

            self.game.move()
            self.set_title('Score: {}'.format(self.game.score))
            pygame.display.update()

        pygame.quit()

    def set_title(self, msg):
        pygame.display.set_caption('Snake AI [' + msg + ']')
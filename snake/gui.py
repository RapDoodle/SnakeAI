import pygame
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

NUM_ROWS = 30
NUM_COLS = 30

GRID_SIZE = 30
GRID_MARGIN = 2
BODY_MARGIN = 30
BODY_BORDER_WIDTH = 2

LEFT = (0, -1)
RIGHT = (0, 1)
UP = (-1, 0)
DOWN = (1, 0)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

MOVE = 0
RAND_FOOD = 1
EAT = 2
SPAWN = 3
END = 4

class SnakeGameGUI():

    def __init__(self, game = None):

        # Register the controller
        self.game = game

        # Color scheme
        self.background_color = BLACK

        self.window_width = NUM_COLS * GRID_SIZE + (NUM_COLS - 1) * GRID_MARGIN + BODY_MARGIN * 2 + BODY_BORDER_WIDTH * 2
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

        x1 = BODY_MARGIN - BODY_BORDER_WIDTH
        y1 = x1
        x2 = self.window_width - 2*(BODY_MARGIN - BODY_BORDER_WIDTH)
        y2 = x2
        pygame.draw.rect(self.screen, WHITE, [x1, y1, x2, y2], BODY_BORDER_WIDTH)

    def draw_block(self, r, c, color):
        x1 = BODY_MARGIN + c*GRID_SIZE + (c+1)*GRID_MARGIN
        y1 = BODY_MARGIN + r*GRID_SIZE + (r+1)*GRID_MARGIN
        pygame.draw.rect(self.screen, color, pygame.Rect([x1, y1, GRID_SIZE, GRID_SIZE]))

    def load(self):
        root = tk.Tk()
        root.withdraw()

        replay_path = filedialog.askopenfilename()

        if len(replay_path) == 0:
            return

        log = pd.read_csv(replay_path)

        self.paint_board()
        self.game.game_init(replay_mode = True)

        for _, row in log.iterrows():
            self.paint_board()
            
            if row['type'] == MOVE or row['type'] == EAT:
                self.game.turn((row['pos_dr'], row['pos_dc']))
                self.game.move(replay_mode = True)
                pygame.time.wait(50)

            elif row['type'] == RAND_FOOD:
                self.game.set_food(row['pos_r'], row['pos_c'])

            elif row['type'] == SPAWN:
                self.game.positions.append((row['pos_r'], row['pos_c']))
                self.game.turn((row['pos_dr'], row['pos_dc']))
                self.game.move(replay_mode = True)

            elif row['type'] == END:
                self.game.end = True

            for pos in self.game.foods:
                self.draw_block(pos[0], pos[1], RED)
                
            for pos in self.game.positions:
                self.draw_block(pos[0], pos[1], WHITE)

            pygame.display.update()
            pygame.event.pump()

    def spin(self):
        clock = pygame.time.Clock()

        self.paint_board()

        while self.running:
            clock.tick(10)
            pygame.event.pump()
            self.paint_board()
            for pos in self.game.foods:
                self.draw_block(pos[0], pos[1], RED)
            for pos in self.game.positions:
                self.draw_block(pos[0], pos[1], WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if self.game is not None:
                        if event.key == pygame.K_UP:
                            self.game.turn(UP)
                        elif event.key == pygame.K_DOWN:
                            self.game.turn(DOWN)
                        elif event.key == pygame.K_LEFT:
                            self.game.turn(LEFT)
                        elif event.key == pygame.K_RIGHT:
                            self.game.turn(RIGHT)
                        elif event.key == pygame.K_r:
                            self.game.game_init()
                        elif event.key == pygame.K_s:
                            self.game.save()
                        elif event.key == pygame.K_l:
                            self.load()

            self.game.move()

            pygame.display.update()

        pygame.quit()
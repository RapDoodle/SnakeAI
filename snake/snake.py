import pandas as pd
import numpy as np
import random
import math
import os

# Grid sizes
num_rows = 30
num_cols = 30

# Directions
LEFT = (0, -1)
RIGHT = (0, 1)
UP = (-1, 0)
DOWN = (1, 0)
DIRECTIONS = [LEFT, RIGHT, UP, DOWN]

MOVE = 0
RAND_FOOD = 1
EAT = 2
SPAWN = 3
END = 4

class SnakeGame():
    
    def __init__(self):
        self.game_init()

    def game_init(self, replay_mode = False):
        # Reduce the time complexity taken to query neural network parameters 
        self.snake_board = np.zeros((num_rows, num_cols), dtype = int)
        self.foods_board = np.zeros((num_rows, num_cols), dtype = int)

        self.log = []
        self.end = False
        self.length = 1
        self.step = 0
        self.positions = []
        self.foods = []
        if not replay_mode:
            init_pos = (num_rows // 2, num_cols // 2)
            self.log.append((self.step, SPAWN, init_pos[0], init_pos[1], 0, 0))
            self.positions.append(init_pos)
            self.snake_board[init_pos[0]][init_pos[1]] = 1
            self.direction = random.choice(DIRECTIONS)
            self.move()
            self.random_foods()
        
    def random_foods(self):
        # Generate a random normal food
        new_food = (np.random.randint(num_rows), np.random.randint(num_cols))
        if new_food in self.positions:
            return self.random_foods()
        self.foods_board[new_food[0]][new_food[1]] = 1
        self.foods.append(new_food)
        self.log.append((self.step, RAND_FOOD, new_food[0], new_food[1], 0, 0))

    def set_food(self, r, c):
        # Used for replay purposes
        new_food = (r, c)
        self.foods.append(new_food)
        
    def get_head_pos(self):
        return self.positions[0]

    def get_tail_pos(self):
        return self.positions[-1]

    def turn(self, direction):
        if self.length > 1 and (direction[0] * -1, direction[1] * -1) == self.direction:
            # Raise invalid direction exception
            pass
        else:
            self.direction = direction

    def move(self, replay_mode = False):
        self.step = self.step + 1
        if self.end == False:
            # Debug code
            # print(self.get_nn_params())
            # End of debug code
            cur_pos = self.get_head_pos()
            d_row, d_col = self.direction
            new = ((cur_pos[0] + d_row), (cur_pos[1] + d_col))
            if (new[0] < 0 or new[0] >= num_rows) or \
                (new[1] < 0 or new[1] >= num_cols) or \
                len(self.positions) > 2 and new in self.positions[2:]:
                print('Dead')
                self.log.append((self.step, END, new[0], new[1], 0, 0))
                self.end = True
            elif new in self.foods:
                # Eat the new food
                self.positions.insert(0, new)
                self.foods.remove(new)
                self.length = self.length + 1
                
                # Set borad states
                self.snake_board[new[0]][new[1]] = 1
                self.foods_board[new[0]][new[1]] = 0
                
                # New food
                if not replay_mode:
                    self.random_foods()

                self.log.append((self.step, EAT, new[0], new[1], self.direction[0], self.direction[1]))
            else:
                self.positions.insert(0, new)
                self.snake_board[new[0]][new[1]] = 1
                if len(self.positions) > self.length:
                    removed = self.positions.pop()
                    self.snake_board[removed[0]][removed[1]] = 0
                self.log.append((self.step, MOVE, new[0], new[1], self.direction[0], self.direction[1]))
    
    def get_nn_params(self):
        # Distance to the wall
        head_r, head_c = self.get_head_pos()

        dtw_n = head_r
        dtw_s = num_rows - 1 - head_r
        dtw_w = head_c
        dtw_e = num_cols - 1 - head_c
        dtw_ne = distance(dtw_n, dtw_e)
        dtw_se = distance(dtw_s, dtw_e)
        dtw_sw = distance(dtw_s, dtw_w)
        dtw_nw = distance(dtw_n, dtw_w)

        params = np.zeros((24,1))
        params[0] = dtw_n
        params[1] = dtw_ne
        params[2] = dtw_e
        params[3] = dtw_se
        params[4] = dtw_s
        params[5] = dtw_sw
        params[6] = dtw_w
        params[7] = dtw_nw

        # Distance to food
        params[8] = get_distance(self.foods_board, head_r, head_c, -1, 0)
        params[9] = get_distance(self.foods_board, head_r, head_c, -1, 1)
        params[10] = get_distance(self.foods_board, head_r, head_c, 0, 1)
        params[11] = get_distance(self.foods_board, head_r, head_c, 1, 1)
        params[12] = get_distance(self.foods_board, head_r, head_c, 1, 0)
        params[13] = get_distance(self.foods_board, head_r, head_c, 1, -1)
        params[14] = get_distance(self.foods_board, head_r, head_c, 0, -1)
        params[15] = get_distance(self.foods_board, head_r, head_c, -1, -1)

        # Distance to itself
        params[16] = get_distance(self.snake_board, head_r, head_c, -1, 0)
        params[17] = get_distance(self.snake_board, head_r, head_c, -1, 1)
        params[18] = get_distance(self.snake_board, head_r, head_c, 0, 1)
        params[19] = get_distance(self.snake_board, head_r, head_c, 1, 1)
        params[20] = get_distance(self.snake_board, head_r, head_c, 1, 0)
        params[21] = get_distance(self.snake_board, head_r, head_c, 1, -1)
        params[22] = get_distance(self.snake_board, head_r, head_c, 0, -1)
        params[23] = get_distance(self.snake_board, head_r, head_c, -1, -1)

        return params

    def save(self, filename = 'default'):
        df = pd.DataFrame(self.log, columns = ['step', 'type', 'pos_r', 'pos_c', 'pos_dr', 'pos_dc'])
        if not os.path.exists('replay'):
            os.makedirs('replay')
        df.to_csv('./replay/{}.csv'.format(filename), index = False)

    def print_board(self):
        print('Snake Board')
        print(self.snake_board + self.foods_board * 2)
        # print('Food Board')
        # print(self.foods_board)

def distance(l1, l2):
    return math.sqrt(math.pow(l1, 2) + math.pow(l2, 2))

def get_distance(board, r, c, dr, dc):
    count = 0
    unit = 1 if dr * dc == 0 else 1.414

    search = True

    while search:
        r = r + dr
        c = c + dc

        # If nothing is found
        if (r < 0 or r >= num_rows) or \
            (c < 0 or c >= num_cols):
            count = 0
            break
        
        count = count + 1

        if board[r][c] == 1:
            break

    return count * unit

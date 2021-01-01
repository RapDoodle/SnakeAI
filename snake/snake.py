import pandas as pd
import numpy as np
import random
import math
import os

from settings import settings
import common.constant as const

class SnakeGame():
    
    def __init__(self):
        self.game_init()

    def game_init(self, replay_mode = False, init_length = 3, init_direction = None, init_pos = None):
        # Reduce the time complexity taken to query neural network parameters 
        self.snake_board = np.zeros((settings['NUM_ROWS'], settings['NUM_COLS']), dtype = int)
        self.foods_board = np.zeros((settings['NUM_ROWS'], settings['NUM_COLS']), dtype = int)
        self.score = 0
        self.fitness = 1
        self.moves_left = settings['MAX_HUNGER']
        self.log = []
        self.end = False
        self.length = init_length
        self.step = 0
        self.positions = []
        self.foods = []
        
        if not replay_mode:
            init_pos = (settings['NUM_ROWS'] // 2, settings['NUM_COLS'] // 2)
            self.direction = random.choice(const.DIRECTIONS)
            self.log.append((self.step, const.SPAWN, init_pos[0], init_pos[1], self.direction[0], self.direction[1]))
            self.positions.append(init_pos)
            self.snake_board[init_pos[0]][init_pos[1]] = 1
            
            inverse_direction = (-1*self.direction[0], -1*self.direction[1])
            new_r = init_pos[0]
            new_c = init_pos[1]
            for _ in range(init_length-1):
                new_r = new_r + inverse_direction[0]
                new_c = new_c + inverse_direction[1]
                self.positions.append((new_r, new_c))

            self.move()
            self.random_foods()
        
    def random_foods(self):
        # Generate a random normal food
        new_food = (np.random.randint(settings['NUM_ROWS']), np.random.randint(settings['NUM_COLS']))
        if new_food in self.positions:
            return self.random_foods()
        self.foods_board[new_food[0]][new_food[1]] = 1
        self.foods.append(new_food)
        self.log.append((self.step, const.RAND_FOOD, new_food[0], new_food[1], 0, 0))

    def turn(self, direction):
        if self.length > 1 and (direction[0] * -1, direction[1] * -1) == self.direction:
            # Raise invalid direction exception
            pass
        else:
            self.direction = direction

    def move(self, replay_mode = False):
        self.step = self.step + 1
        self.moves_left = self.moves_left - 1

        if self.end == False:
            # Debug code
            # print(self.get_nn_params())
            # End of debug code
            cur_pos = self.get_head_pos()
            d_row, d_col = self.direction
            new = ((cur_pos[0] + d_row), (cur_pos[1] + d_col))

            move_type = 'None'
            pos_r = cur_pos[0]
            pos_c = cur_pos[0]
            pos_dr = 'None'
            pos_dc = 'None'

            if (new[0] < 0 or new[0] >= settings['NUM_ROWS']) or \
                (new[1] < 0 or new[1] >= settings['NUM_COLS']) or \
                len(self.positions) > 2 and new in self.positions[2:]:
                # print('Dead')
                move_type = const.END
                pos_dr = 0
                pos_dc = 0
                
                self.end = True
            elif new in self.foods:
                # Eat the new food
                self.positions.insert(0, new)
                self.foods.remove(new)
                self.length = self.length + 1

                # Reset moves left
                self.moves_left = settings['MAX_HUNGER']
                
                # Set borad states
                self.snake_board[new[0]][new[1]] = 1
                self.foods_board[new[0]][new[1]] = 0
                
                # New food
                if not replay_mode:
                    self.random_foods()
                
                move_type = const.EAT
                pos_dr = self.direction[0]
                pos_dc = self.direction[1]

                self.score = self.score + 1

            else:
                self.positions.insert(0, new)
                self.snake_board[new[0]][new[1]] = 1
                if len(self.positions) > self.length:
                    removed = self.positions.pop()
                    self.snake_board[removed[0]][removed[1]] = 0
                
                move_type = const.MOVE
                pos_dr = self.direction[0]
                pos_dc = self.direction[1]

            if self.moves_left <= 0:
                move_type = const.END

            self.fitness = self.get_fitness()
            self.log.append((self.step, move_type, pos_r, pos_c, pos_dr, pos_dc))
            
            return move_type

    def get_nn_params(self):
        # Distance to the wall
        head_r, head_c = self.get_head_pos()

        dtw_n = head_r
        dtw_s = settings['NUM_ROWS'] - 1 - head_r
        dtw_w = head_c
        dtw_e = settings['NUM_COLS'] - 1 - head_c
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

        # Use binary vision
        params = np.where(params > 0, 1, 0)

        return params

    def get_fitness(self):
        # return 300 + 5 * self.score - self.moves_left
        return self.score

    def set_food(self, r, c):
        # Used for replay purposes
        new_food = (r, c)
        self.foods.append(new_food)
        
    def get_head_pos(self):
        return self.positions[0]

    def get_tail_pos(self):
        return self.positions[-1]

    def up(self):
        self.turn(UP)

    def down(self):
        self.turn(DOWN)

    def left(self):
        self.turn(LEFT)

    def right(self):
        self.turn(RIGHT) 

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
        if (r < 0 or r >= settings['NUM_ROWS']) or \
            (c < 0 or c >= settings['NUM_COLS']):
            count = 0
            break
        
        count = count + 1

        if board[r][c] == 1:
            break

    return count * unit

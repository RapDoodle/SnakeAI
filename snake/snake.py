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

    def game_init(self, replay_mode = False, init_len = settings['INIT_LEN'], init_direction = None, init_pos = None):
        # Reduce the time complexity taken to query neural network parameters 
        self.snake_board = np.zeros((settings['NUM_ROWS'], settings['NUM_COLS']), dtype = int)
        self.apple_board = np.zeros((settings['NUM_ROWS'], settings['NUM_COLS']), dtype = int)
        self.banana_board = np.zeros((settings['NUM_ROWS'], settings['NUM_COLS']), dtype = int)

        # Game status
        self.positions = []
        self.apples = []
        self.bananas = []
        self.step = 0
        self.score = 0
        self.fitness = 1
        self.moves_left = settings['MAX_HUNGER']
        self.end = False
        self.length = init_len

        # Logging info
        self.log = []
        
        if not replay_mode:
            # Set the initial position and pick a random position
            init_pos = (settings['NUM_ROWS'] // 2, settings['NUM_COLS'] // 2)
            self.direction = random.choice(const.DIRECTIONS)
            self.positions.append(init_pos)
            self.snake_board[init_pos[0]][init_pos[1]] = 1

            # @LOG: SPANW INFO
            self.log.append((self.step, const.SPAWN, init_pos[0], init_pos[1], self.direction[0], self.direction[1]))
            
            # Generate the body of the remaining snake
            inverse_direction = (-1*self.direction[0], -1*self.direction[1])
            new_r = init_pos[0]
            new_c = init_pos[1]
            for _ in range(init_len-1):
                new_r = new_r + inverse_direction[0]
                new_c = new_c + inverse_direction[1]
                self.positions.append((new_r, new_c))

            # Force the snake to move
            self.random_apple()
            self.move()
        
    def random_apple(self):
        # Generate a random normal food (apple)
        new_apple = (np.random.randint(settings['NUM_ROWS']), np.random.randint(settings['NUM_COLS']))
        if new_apple in self.positions:
            return self.random_apple()
        self.apple_board[new_apple[0]][new_apple[1]] = 1
        self.apples.append(new_apple)

        # @LOG: New apple
        self.log.append((self.step, const.RAND_APPLE, new_apple[0], new_apple[1], 0, 0))

    def random_banans(self, depth = settings['NUM_ROWS'] * settings['NUM_COLS']):
        # Generate a random banana
        new_banana = (np.random.randint(settings['NUM_ROWS']), np.random.randint(settings['NUM_COLS']))
        if new_banana in self.positions or self.apple_board[new_banana[0]][new_banana[1]] != 0:
            return self.random_banans()
        self.banana_board[new_banana[0]][new_banana[1]] = 1
        self.bananas.append(new_banana)

        # @LOG: New banana
        self.log.append((self.step, const.RAND_BANANA, new_banana[0], new_banana[1], 0, 0))

    def turn(self, direction):
        if self.length > 1 and (direction[0] * -1, direction[1] * -1) == self.direction:
            # Invalid direction
            return
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
                move_type = const.END
                pos_dr = 0
                pos_dc = 0
                
                self.end = True
            elif new in self.apples:
                # Eat the apple
                self.positions.insert(0, new)
                self.apples.remove(new)

                self.length = self.length + 1

                # Reset moves left
                self.moves_left = settings['MAX_HUNGER']
                
                # Set borad states
                self.snake_board[new[0]][new[1]] = 1
                self.apple_board[new[0]][new[1]] = 0
                
                # New food
                if not replay_mode:
                    self.random_apple()
                    # Bananas can only be generated after eating apples
                    if np.random.rand() <= settings['BANANA_GENERATION_RATE']:
                        self.random_banans()
                
                move_type = const.EAT
                pos_dr = self.direction[0]
                pos_dc = self.direction[1]

                self.score = self.score + 1

            elif new in self.bananas:
                # Eat a banana
                self.positions.insert(0, new)
                self.bananas.remove(new)

                self.length = self.length + 1

                # Reset moves left
                self.moves_left = settings['MAX_HUNGER']
                
                # Set borad states
                self.snake_board[new[0]][new[1]] = 1
                self.banana_board[new[0]][new[1]] = 0
                
                move_type = const.EAT
                pos_dr = self.direction[0]
                pos_dc = self.direction[1]

                self.score = self.score + 5
            else:  # Moving
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

            # @LOG: Move
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

        params = np.zeros((32,1))
        params[0] = dtw_n
        params[1] = dtw_ne
        params[2] = dtw_e
        params[3] = dtw_se
        params[4] = dtw_s
        params[5] = dtw_sw
        params[6] = dtw_w
        params[7] = dtw_nw

        # Distance to apple
        params[8] = get_distance(self.apple_board, head_r, head_c, -1, 0)
        params[9] = get_distance(self.apple_board, head_r, head_c, -1, 1)
        params[10] = get_distance(self.apple_board, head_r, head_c, 0, 1)
        params[11] = get_distance(self.apple_board, head_r, head_c, 1, 1)
        params[12] = get_distance(self.apple_board, head_r, head_c, 1, 0)
        params[13] = get_distance(self.apple_board, head_r, head_c, 1, -1)
        params[14] = get_distance(self.apple_board, head_r, head_c, 0, -1)
        params[15] = get_distance(self.apple_board, head_r, head_c, -1, -1)

        # Distance to banana
        params[16] = get_distance(self.banana_board, head_r, head_c, -1, 0)
        params[17] = get_distance(self.banana_board, head_r, head_c, -1, 1)
        params[18] = get_distance(self.banana_board, head_r, head_c, 0, 1)
        params[19] = get_distance(self.banana_board, head_r, head_c, 1, 1)
        params[20] = get_distance(self.banana_board, head_r, head_c, 1, 0)
        params[21] = get_distance(self.banana_board, head_r, head_c, 1, -1)
        params[22] = get_distance(self.banana_board, head_r, head_c, 0, -1)
        params[23] = get_distance(self.banana_board, head_r, head_c, -1, -1)

        # Distance to itself
        params[24] = get_distance(self.snake_board, head_r, head_c, -1, 0)
        params[25] = get_distance(self.snake_board, head_r, head_c, -1, 1)
        params[26] = get_distance(self.snake_board, head_r, head_c, 0, 1)
        params[27] = get_distance(self.snake_board, head_r, head_c, 1, 1)
        params[28] = get_distance(self.snake_board, head_r, head_c, 1, 0)
        params[29] = get_distance(self.snake_board, head_r, head_c, 1, -1)
        params[30] = get_distance(self.snake_board, head_r, head_c, 0, -1)
        params[31] = get_distance(self.snake_board, head_r, head_c, -1, -1)

        # Use binary vision
        params = np.where(params > 0, 1, 0)

        return params

    def get_fitness(self):
        # return 300 + 5 * self.score - self.moves_left
        return self.score

    def set_apple(self, r, c):
        # Used for replay purposes
        new_apple = (r, c)
        self.apples.append(new_apple)
        self.apple_board[r][c] = 1
    
    def set_banana(self, r, c):
        # Used for replay purposes
        new_banana = (r, c)
        self.bananas.append(new_banana)
        self.banana_board[r][c] = 1
        
    def get_head_pos(self):
        return self.positions[0]

    def get_tail_pos(self):
        return self.positions[-1]

    def up(self):
        self.turn(const.UP)

    def down(self):
        self.turn(const.DOWN)

    def left(self):
        self.turn(const.LEFT)

    def right(self):
        self.turn(const.RIGHT) 

    def save(self, filename = 'default'):
        df = pd.DataFrame(self.log, columns = ['step', 'type', 'pos_r', 'pos_c', 'pos_dr', 'pos_dc'])
        if not os.path.exists('replay'):
            os.makedirs('replay')
        try:
            df.to_csv('./replay/{}.csv'.format(filename), index = False)
        except:
            print('[WARNING] Access denied. Unable to write.')

    def print_board(self):
        # Print the board using Numpy. For debugging purposes only
        print('Snake Board')
        print(self.snake_board + self.apple_board * 2)
        # print('Food Board')
        # print(self.apple_board)

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

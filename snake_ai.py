import os
import numpy as np
import pandas as pd
from datetime import datetime

import common.constant as const
from nn.genetic_algorithm.layers import GAInput, GADense, GAOutput
from nn.genetic_algorithm.model import GAModel, GASequentialModel
from snake.snake import SnakeGame
from common.utils import save_object

model = GAModel(population = 500)
now = datetime.now()
t_str = now.strftime("%Y-%m-%d-%H-%M-%S")

model.add(GAInput(32))
model.add(GADense(24, activation = 'tanh', use_bias = True, kernel_initializer = 'uniform'))
model.add(GADense(18, activation = 'tanh', use_bias = True, kernel_initializer = 'uniform'))
model.add(GAOutput(4, activation = 'softmax', use_bias = False))

model.new_population()

df = pd.DataFrame(columns = ['iteration', 'min_score', 'max_score'])

def play_snake(model, params = (False, 'untitled')):
    """Params: (save_game, name)
        save_game: bool
        name: str
    """
    game = SnakeGame()

    # Register the game in GAModel
    model.register = game
    
    end = False

    while not end:
        X_pred = game.get_nn_params()
        y_pred = model.predict(X_pred)
        choice = y_pred.argmax()

        if choice == 0:
            game.up()
        elif choice == 1:
            game.down()
        elif choice == 2:
            game.left()
        else:
            game.right()

        result = game.move()

        if result == const.END:
            break

    # Set reward
    model.set_reward(game.fitness)

    # Save game
    if params[0] == True:
        if params[1] is None:
            raise Exception('name not specified')
        game.save(params[1])

iters = 10
for i in range(iters):
    print('Iteration: {}/{}'.format(i+1, iters))
    min_reward, max_reward = model.simulate(play_snake, keep_rate=0.6, mutate_rate=0.01, params = (False,))
    model.forest[0].register.save('{}-{}-{}'.format(t_str, i+1, model.forest[0].reward))

    # Log to statistics
    df.at[i, 'iteration'] = i+1
    df.at[i, 'min_score'] = min_reward
    df.at[i, 'max_score'] = max_reward

    if i % 1 == 0:
        try:
            if not os.path.exists('statistics'):
                os.makedirs('statistics')
            df.to_csv('./statistics/{}.csv'.format(t_str))
        except:
            pass

if not os.path.exists('models'):
    os.makedirs('models') 

save_object(model, './models/{}.obj'.format(t_str))
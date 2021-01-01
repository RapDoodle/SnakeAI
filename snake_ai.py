import numpy as np
import common.constant as const
from nn.genetic_algorithm.layers import GAInput, GADense, GAOutput
from nn.genetic_algorithm.model import GAModel, GASequentialModel
from snake.snake import SnakeGame

model = GAModel(population = 200)

model.add(GAInput(24))
model.add(GADense(18, activation = 'tanh', use_bias = True, kernel_initializer = 'uniform'))
model.add(GADense(18, activation = 'tanh', use_bias = True, kernel_initializer = 'uniform'))
model.add(GAOutput(4, activation = 'softmax', use_bias = False))

model.new_population()

MOVE = 0
RAND_FOOD = 1
EAT = 2
SPAWN = 3
END = 4

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

        if result == END:
            break

    # Set reward
    model.set_reward(game.fitness)

    # Save game
    if params[0] == True:
        if params[1] is None:
            raise Exception('name not specified')
        game.save(params[1])


for i in range(1000):
    model.simulate(play_snake, keep_rate=0.6, mutate_rate=0.01, params = (False,))
    model.forest[0].register.save('FORMAL-{}-{}'.format(i+1, model.forest[0].reward))
from snake.gui import SnakeGameGUI
from snake.snake import SnakeGame
snake_game = SnakeGame()
gui = SnakeGameGUI(snake_game)
gui.spin()
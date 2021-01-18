# SnakeAI

Snake game is a computer action game whose goal is to eat as much food as possible. In this project, we will be training a Snake AI using a neural network with a genetic algorithm and trying to understand some of the model’s behaviors.
For a video demo of the project, [click here (YouTube)](https://youtu.be/y_lS4VRrUio)

<img src="https://img.youtube.com/vi/y_lS4VRrUio/maxresdefault.jpg" height="350"/>

## Current progress
- The project has served its purpose as a course project.
- The general-purpose neural network built for the project will become an independent project on GitHub. For more information about the project LightNeuNet, click [here](https://github.com/RapDoodle/LightNeuNet).
- I will continue to explore the process of parameter tuning to see if it could yield better results.

## Usage
### Install dependencies
Make sure NumPy and PyGame is installed
```bash
pip install numpy
pip install pygame
```

### Train the neural network
```python snake_ai.py```

### Play the game (user control)
```python snake_ai.py```

### Adjust the neural network
Edit ```snake_ai.py```

### Adjust more parameters
Edit ```settings.py```

## Components in the Repository

### The Snake Game
Built on top of PyGame. Supports full user control, game replay, and exposes APIs for neural networks.

### LightNeuNet
LightNeuNet is a light-weight Neural Network framework built for academic use. Since the author (me) is only a Y3 student, the framework should only be used for academic and hilarious purposes. Please do not use it in production yet.

## Detailed Introduction
Snake is a computer action game. Typically, players control the game by using four buttons: up, down, left, and right. By clicking the buttons, it determines the direction of the snake’s movement. The goal of the snake is to eat as much food as possible to grow. The game is won by filling up all the grids in the game. If the snake hits the wall or runs into itself without filling up all the grids, the snake dies, and the game ends. When the snake eats an apple, it will be longer by one grid. So, players need to control the snake to eat as much food as possible while avoiding hitting the wall or running into itself.

In our approach, we built an AI agent that learned to play the game from knowing nothing to making some intelligent decisions. To help the AI learn by itself, the snake is equipped with binary visions in eight directions; it can identify where the food is, where the wall is, and whether it is a part of itself.

There are many papers on the topic, but most of them only focused on the variant of snake games that only contain one type of food at a time. Our project focused on the behavior of the neural network for more than one type of food. We modified the original game is the following ways:

1. The game board size is 15 × 15
1. There are two types of food, apples and bananas.
1. There will always be an apple. But bananas are generated at random upon the snake eating an apple.
1. Upon eating an apple, there will be a 15% chance of generating a new banana.
1. There could be multiple bananas on the board. We deem it as a strategy to seduce the snake to eat the bananas.
1. Eating an apple adds one point to the score while eating a banana will add five points.
1. Eating both types of food will add a length of one to the snake.

We are interested in whether the model could learn to prioritize the bananas since it adds more reward to the total score. In this project, we implemented the model using a neural network with a genetic algorithm.

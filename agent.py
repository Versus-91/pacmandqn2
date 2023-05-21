import pandas as pd
import numpy as np
from constants import *

from game import GameWrapper

raw_maze_data = []
with open('maze1.txt', 'r') as f:
    for line in f:
        raw_maze_data.append(line.split())
maze_data = np.array(raw_maze_data)
for idx, values in enumerate(maze_data):
    for id, value in enumerate(values):
        if value == '.' or value == 'p' or value == '+':
            maze_data[idx][id] = 1
        else:
            maze_data[idx][id] = 0
maze_data = maze_data.astype(dtype=np.float32)
print(maze_data.shape)
game = GameWrapper()
game.start()
prev_x = 0
prev_y = 0
index = 0
while True:
    # game.step(UP)
    game.update()
    x = int(game.pacman_position().x / 16) 
    y = int(game.pacman_position().y / 16) 
    if x == prev_x and y == prev_y:
        continue
    prev_x = x 
    prev_y = y 
    if maze_data[int(y)][int(x)] == 1:
        index += 1
        print("maze value ="+ str(index),maze_data[int(y)][int(x)])
        print(x, y)
    else: 
        print("maze index =", x,y)


from constants import *
from run import GameController
import matplotlib.pyplot as plt

game = GameController()
game.startGame()
while True:
    state = game.perform_action(UP)
    image = state[0].swapaxes(0, 1)
    plt.imshow(image)
    plt.show()
    print(state)

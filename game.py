from constants import *
from run import GameController
import matplotlib.pyplot as plt


class GameWrapper:
    def __init__(self):
        self.controller = GameController()
        self.action = UP

    def reset(self):
        self.controller.startGame()

    def step(self, action):
        return self.controller.perform_action(action)


if __name__ == "__main__":
    controller = GameWrapper()
    controller.reset()
    while True:
        state = controller.step(UP)
        image = state[0].swapaxes(0, 1)
        plt.imshow(image)
        plt.show()

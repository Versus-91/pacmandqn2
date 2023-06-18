import cv2
import numpy as np
from constants import *
from run import GameController
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


class GameWrapper:
    def __init__(self):
        self.controller = GameController()
        self.action = UP

    def start(self):
        self.controller.startGame()

    def restart(self):
        self.controller.restartGame()

    def step(self, action):
        assert action >= 0 and action < 4
        if action == 0:
            action = UP
        elif action == 1:
            action = DOWN
        elif action == 2:
            action = LEFT
        elif action == 3:
            action = RIGHT
        else:
            print("Invalid action", action)
        data = self.controller.perform_action(action)
        return (data[0], data[1], data[2], data[3])

    def pacman_position(self):
        return self.controller.pacman.position

    def update(self):
        self.controller.update()

    def process_image(self, obs):
        # image = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (210, 160))
        # image = np.array(image, dtype=np.float32) / 255.0
        return obs


if __name__ == "__main__":
    controller = GameWrapper()
    controller.start()
    while True:
        state = controller.step(UP)
        print(controller.pacman_position())

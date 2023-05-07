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

    def reset(self):
        self.controller.startGame()
        return self.step(RIGHT)

    def step(self, action):
        data = self.controller.perform_action(action)
        image = data[0].swapaxes(0, 1)
        return (self.process_image(image), data[1], data[2], data[3])

    def process_image(self, obs):
        # image = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (210, 160))
        # image = np.array(image, dtype=np.float32) / 255.0
        return obs


if __name__ == "__main__":
    controller = GameWrapper()
    controller.reset()
    while True:
        state = controller.step(UP)

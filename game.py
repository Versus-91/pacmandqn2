import cv2
import numpy as np
from constants import *
from run import GameController
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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
        gray_img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        width = 300
        height = 300
        dim = (width, height)
        # resize image
        resized = cv2.resize(gray_img, dim, interpolation=cv2.INTER_AREA)
        return resized


if __name__ == "__main__":
    controller = GameWrapper()
    controller.reset()
    while True:
        state = controller.step(UP)

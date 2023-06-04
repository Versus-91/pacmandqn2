import cv2
from matplotlib import pyplot as plt
import numpy as np
import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData

game_states = {
    'pallet': 1,
    'powerpallet': 0.9,
    'ghost_fright': 0.6,
    'ghost': 0.5,
    'pacman': 0.4,
    'wall': 0.1,
    'path': 0.2,
    'empty': 0
}


class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(False)
        self.level = 0
        self.lives = 3
        self.score = 0
        self.last_score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(
            self.background_norm, self.level % 5)
        self.background_flash = self.mazesprites.constructBackground(
            self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):
        self.textgroup.hideText()
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(
            self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(
            *self.mazedata.obj.pacmanStart))
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
        self.eatenPellets = []
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        self.ghosts.pinky.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(
            *self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.ghosts.pinky.startNode.denyAccess(RIGHT, self.ghosts.pinky)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

    def get_sate_old(self):
        raw_maze_data = []
        with open('maze1.txt', 'r') as f:
            for line in f:
                raw_maze_data.append(line.split())
        maze_data = np.array(raw_maze_data)
        pellets = np.zeros(maze_data.shape)
        ghosts = np.zeros(maze_data.shape)
        powerpellets = np.zeros(maze_data.shape)
        walls = np.zeros(maze_data.shape)
        for idx, values in enumerate(maze_data):
            for id, value in enumerate(values):
                # if value == '.' or value == 'p' or value == '+':
                # pallets.push((idx, id))
                if value == 'X':
                    walls[idx][id] = game_states.get('empty')
                if value in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                    walls[idx][id] = game_states.get('wall')
                if value == 'n' or value == '|' or value == '-':
                    walls[idx][id] = game_states.get('path')
        for idx, values in enumerate(self.eatenPellets):
            x = int(values.position.x / 16)
            y = int(values.position.y / 16)
            walls[y][x] = game_states.get('path')
        for idx, values in enumerate(self.pellets.pelletList):
            x = int(values.position.x / 16)
            y = int(values.position.y / 16)
            walls[y][x] = game_states.get('pallet')
        for idx, values in enumerate(self.pellets.powerpellets):
            pellets[y][x] = game_states.get('powerpallet')

        x = int(self.pacman.position.x / 16)
        y = int(self.pacman.position.y / 16)
        walls[y][x] = game_states.get('pacman')
        x = int(self.ghosts.blinky.position.x / 16)
        y = int(self.ghosts.blinky.position.y / 16)
        if self.ghosts.blinky.mode.current is not FREIGHT:
            walls[y][x] = game_states.get('ghost')
        else:
            walls[y][x] = game_states.get('ghost_fright')

        x = int(self.ghosts.inky.position.x / 16)
        y = int(self.ghosts.inky.position.y / 16)
        if self.ghosts.inky.mode.current is not FREIGHT:
            walls[y][x] = game_states.get('ghost')
        else:
            walls[y][x] = game_states.get('ghost_fright')
        x = int(self.ghosts.pinky.position.x / 16)
        y = int(self.ghosts.pinky.position.y / 16)
        if self.ghosts.pinky.mode.current is not FREIGHT:
            walls[y][x] = game_states.get('ghost')
        else:
            walls[y][x] = game_states.get('ghost_fright')
        x = int(self.ghosts.clyde.position.x / 16)
        y = int(self.ghosts.clyde.position.y / 16)
        if self.ghosts.clyde.mode.current is not FREIGHT:
            walls[y][x] = game_states.get('ghost')
        else:
            walls[y][x] = game_states.get('ghost_fright')

        # state.append(maze_data)
        # state.append(ghosts_position)
        # state.append(self.pacman.direction)
        # state.append(pellets_position)
        walls = walls[7:29, 5:23]
        pellets = pellets[7:29, 5:23]
        ghosts = ghosts[7:29, 5:23]
        return [walls, pellets, ghosts]

    def startGame_old(self):
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0, 17), (27, 17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12, 14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15, 14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(
            self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(
            self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(
            self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(
            self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)

    def update(self):
        dt = self.clock.tick(60) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt)
        else:
            self.pacman.update(dt)
        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()
        self.get_sate()

    def perform_action(self, action):
        invalid_move = False
        if not self.pacman.validDirection(action):
            invalid_move = True
        dt = self.clock.tick(60) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt, action)
        else:
            self.pacman.update(dt)
        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()

        surface_array = pygame.surfarray.array3d(pygame.display.get_surface())
        return (self.get_sate(), self.score, self.lives == 0 or (self.pellets.isEmpty()), self.lives, invalid_move)

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.textgroup.hideText()
                            self.hideEntities()

    def check_ghost_pos(seld, wall, x, y):
        if (x >= 11 and x <= 16) and (y >= 15 and y <= 18):
            return
        # else:
            # assert wall != game_states.get('wall'), f"{x} - {y} are in wall"

    def get_sate(self):
        raw_maze_data = []
        with open('maze1.txt', 'r') as f:
            for line in f:
                raw_maze_data.append(line.split())
        maze_data = np.array(raw_maze_data)
        pellets = np.zeros(maze_data.shape)
        ghosts = np.zeros(maze_data.shape)
        frightened_ghosts = np.zeros(maze_data.shape)
        pacman = np.zeros(maze_data.shape)
        powerpellets = np.zeros(maze_data.shape)
        walls = np.zeros(maze_data.shape)
        for idx, values in enumerate(maze_data):
            for id, value in enumerate(values):
                # if value == '.' or value == 'p' or value == '+':
                # pallets.push((idx, id))
                if value in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '=', 'X']:
                    walls[idx][id] = game_states.get('wall')
                if value == 'n' or value == '|' or value == '-' or value == '.' or value == 'p' or value == '+':
                    walls[idx][id] = game_states.get('path')
        for idx, values in enumerate(self.eatenPellets):
            x = int(values.position.x / 16)
            y = int(values.position.y / 16)
            walls[y][x] = game_states.get('path')
        for idx, values in enumerate(self.pellets.pelletList):
            x = int(values.position.x / 16)
            y = int(values.position.y / 16)
            pellets[y][x] = game_states.get('pallet')
        for idx, powepellet in enumerate(self.pellets.powerpellets):
            x = int(powepellet.position.x / 16)
            y = int(powepellet.position.y / 16)
        x = int(round(self.pacman.position.x / 16))
        y = int(round(self.pacman.position.y / 16))
        pacman[y][x] = game_states.get('pacman')
        assert walls[y][x] != game_states.get('wall')
        x = int(round(self.ghosts.blinky.position.x / 16))
        y = int(round(self.ghosts.blinky.position.y / 16))
        self.check_ghost_pos(walls[y][x], x, y)
        if self.ghosts.blinky.mode.current is not FREIGHT:
            ghosts[y][x] = game_states.get('ghost')
        else:
            frightened_ghosts[y][x] = game_states.get('ghost_fright')

        x = int(round(self.ghosts.inky.position.x / 16))
        y = int(round(self.ghosts.inky.position.y / 16))
        self.check_ghost_pos(walls[y][x], x, y)
        if self.ghosts.inky.mode.current is not FREIGHT:
            ghosts[y][x] = game_states.get('ghost')
        else:
            frightened_ghosts[y][x] = game_states.get('ghost_fright')

        x = int(round(self.ghosts.pinky.position.x / 16))
        y = int(round(self.ghosts.pinky.position.y / 16))
        self.check_ghost_pos(walls[y][x], x, y)
        if self.ghosts.pinky.mode.current is not FREIGHT:
            ghosts[y][x] = game_states.get('ghost')
        else:
            frightened_ghosts[y][x] = game_states.get('ghost_fright')
        x = int(round(self.ghosts.clyde.position.x / 16))
        y = int(round(self.ghosts.clyde.position.y / 16))
        self.check_ghost_pos(walls[y][x], x, y)
        if self.ghosts.clyde.mode.current is not FREIGHT:
            ghosts[y][x] = game_states.get('ghost')
        else:
            frightened_ghosts[y][x] = game_states.get('ghost_fright')
        # state.append(maze_data)
        # state.append(ghosts_position)
        # state.append(self.pacman.direction)
        # state.append(pellets_position)
        # walls = walls[7:29, 5:23]
        # pellets = pellets[7:29, 5:23]
        # ghosts = ghosts[7:29, 5:23]
        return [walls[7:29, 5:23], pacman[7:29, 5:23], pellets[7:29, 5:23], powerpellets[7:29, 5:23], ghosts[7:29, 5:23], frightened_ghosts[7:29, 5:23]]

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.eatenPellets.append(pellet)
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            # if self.pellets.numEaten == 30:
            #     self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            # if self.pellets.numEaten == 70:
            #     self.ghosts.clyde.startNode.allowAccess(
            #         LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(
                        str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=0.1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(
                                pauseTime=0.1, func=self.restartGame)
                        else:
                            self.pause.setPause(
                                pauseTime=0.1, func=self.resetLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(
                    self.nodes.getNodeFromTiles(9, 20), self.level)
                print(self.fruit)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(
                    self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = False
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.lives = 3
        self.level = 0
        self.pause.paused = False
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.textgroup.hideText()
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        self.last_score = 0
        self.pause.paused = False
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        # self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        pygame.display.update()


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()

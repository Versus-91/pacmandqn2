
class GameState:
    def __init__(self):
        self.lives = 0
        self.invalid_move = False
        self.total_pellets = 0
        self.collected_pellets = 0
        self.frame = []
        self.food_distance = -1
        self.ghost_distance = -1
        self.scared_ghost_distance = -1
        self.powerup_distance = -1

from collections import deque, namedtuple
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

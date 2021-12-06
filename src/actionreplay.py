import random

class ActionReplay:
    def __init__(self):
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

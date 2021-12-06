import gym
import numpy as np
import random
import math
import agent

class GameEnv:
    def __init__(self, envi):
        self.env = gym.make(envi)

    def grayscale(s):
        # np.dot(s[...,:3], [0.299,0.587, 0.144])
        last_axis = -1
        s = np.mean(s, axis=2).astype(np.uint8)
        return np.expand_dims(s, last_axis)

    def downsample(s):
        return s[::2, ::2]

    def preprocess(s):
        return GameEnv.grayscale(GameEnv.downsample(s))

    def run(self, Agent):
        s = self.env.reset()
        # print("s, before", s.shape)
        # s = Environment.preprocess(s)
        # print("s, after", s.shape)
        reward = 0
        while True:
            self.env.render()
            a = Agent.act(s)
            frame, r, is_done, info = self.env.step(a)

            if is_done:
                frame = None

            Agent.observe_sample((s, a, r, frame))
            Agent.replay()
            s = frame
            reward += r

            if is_done:
                break
        print("Total reward:", reward)

import gym
import time
import numpy as np
import random
import math
import brain
import actionreplay

MAX_EPSILON = 1
MIN_EPSILON = 0.001
LAMBDA = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
EPOCHS = 1
class Agent:
     epsilon = MAX_EPSILON
     time = 0
     def __init__(self, actionCount, stateCount):
         self.actionCount = actionCount
         self.stateCount = stateCount
         self.brain = brain.Brain(actionCount, stateCount)
         self.memory = actionreplay.ActionReplay()

     def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount-1)
        else:
            return np.argmax(self.brain.predictOneStep(s))

     def observe_sample(self, sample):
        self.memory.add(sample)
        self.time += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.time)

     def replay(self):
        batchSamples = self.memory.sample(BATCH_SIZE)
        no_state = np.zeros((self.stateCount, 160, 3))
        states = np.array([i[0] for i in batchSamples])
        states_  = np.array([(no_state if i[3] is None else i[3]) for i in batchSamples])
        print("states", states.shape)
        prediction = self.brain.predict(states)
        prediction_ = self.brain.predict(states_)
        print("batch", len(batchSamples))
        x = np.zeros((len(batchSamples), self.stateCount, 160, 3))
        y = np.zeros((len(batchSamples), self.actionCount))
        print("x", x.shape)
        print("y", y.shape)

        for j in range(len(batchSamples)):
            batch = batchSamples[j]
            s = batch[0]
            a = batch[1]
            r = batch[2]
            s_ = batch[3]
            t = prediction[j]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(prediction_[j])
            x[j] = s
            y[j] = t
        self.brain.train(x, y, BATCH_SIZE, EPOCHS)

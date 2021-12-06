import gym
import numpy as np
import gameenv
import agent
if __name__ == '__main__':
    game = 'BreakoutDeterministic-v0'
    env = gameenv.GameEnv(game)
    stateCount = env.env.observation_space.shape[0]
    actionCount = env.env.action_space.n
    print(stateCount)
    print(actionCount)
    agent = agent.Agent(actionCount, stateCount)

    while True:
        env.run(agent)

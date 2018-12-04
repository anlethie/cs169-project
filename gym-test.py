# Copied from gym.openai.com/docs/
# on 11/22/18

import gym

if __name__ == '__main__':
    env = gym.make('Pong-ram-v0')
    for i_episode in range(1):
        observation = env.reset()
        for t in range(10000):
            env.render()
            print(observation)
            action = 4
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
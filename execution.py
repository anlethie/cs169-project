import time

def simulate(actor, environment, max_steps=1000, render=False, fps=30):
    """Executes a full episode of actor in environment, for at most max_steps time steps
If render is True, will render simulation on screen at ~fps frames per second."""
    total_reward = 0
    observation  = environment.reset()
    frame_delay  = 1.0 / fps # actually need seconds per frame for sleep method
    done         = False
    steps        = 0
    while not done and steps < max_steps:
        if render:
            environment.render()
            time.sleep(frame_delay)
        action = actor.react_to(observation)
        observation, reward, done, info = environment.step(action)
        steps        += 1
        total_reward += reward

    return total_reward
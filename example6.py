import gym
from actor import GeneticPerceptronActor
from execution import simulate
from genetics import evolve

print('Building environment...', flush=True)
env = gym.make('MsPacman-ram-v0')
print('Generating population...', flush=True)
population = [GeneticPerceptronActor(env.observation_space, env.action_space) for _ in range(100)]
print('Evolving...', flush=True)
evolve(population, env, generations=101, simulation_reps=5, max_steps=20000, render_gens=10, allow_parallel=False)
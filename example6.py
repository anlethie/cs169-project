import gym
from actor import GeneticPerceptronActor
from execution import simulate
from genetics import evolve

print('Building environment...', flush=True)
env = gym.make('MsPacman-ram-v0')
print('Generating population...', flush=True)
population = [GeneticPerceptronActor(env.observation_space, env.action_space) for _ in range(5)]
print('Evolving...', flush=True)
evolve(population, env, generations=50, simulation_reps=1, max_steps=5000, render_gens='BW', allow_parallel=False,fps=100, p_mutation=.01)
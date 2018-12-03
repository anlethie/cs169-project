import gym
from actor import GeneticNNActor
from execution import simulate
from genetics import evolve

HIDDEN_LAYERS = [10, 4]

env = gym.make('MsPacman-ram-v0')
population = [GeneticNNActor(env.observation_space, env.action_space, hidden_layers=HIDDEN_LAYERS) for _ in range(100)]
evolve(population, env, generations=101, simulation_reps=5, max_steps=20000, render_gens=5, allow_parallel=True)
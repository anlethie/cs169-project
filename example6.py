import gym
from actor import GeneticNNActor
from genetics import evolve, render_from_file

<<<<<<< HEAD
print('Building environment...', flush=True)
env = gym.make('MsPacman-ram-v0')
print('Generating population...', flush=True)
population = [GeneticPerceptronActor(env.observation_space, env.action_space) for _ in range(5)]
print('Evolving...', flush=True)
evolve(population, env, generations=50, simulation_reps=1, max_steps=5000, render_gens='BW', allow_parallel=False,fps=100, p_mutation=.01)
=======
HIDDEN_LAYERS = [10, 6]
SAVEFILE      = 'MsPacman_NN_10_6.txt'

try:
    env = gym.make('MsPacman-ram-v0')
    population = [GeneticNNActor(env.observation_space, env.action_space, hidden_layers=HIDDEN_LAYERS) for _ in range(100)]
    model = population[0]
    evolve(population, env,
        generations=1000, simulation_reps=5,
        p_mutation=0.05, mutation_scale=0.25,
        max_steps=100000, render_gens=None,
        savefile=SAVEFILE,
        savenum=3,
        allow_parallel=True
        )
except KeyboardInterrupt:
    print('Interrupted...')
    pass
finally:
    print('Top 3 Actors found:')
    render_from_file(SAVEFILE, model, env, num=3)
>>>>>>> 2ff9d269724ee00f0c737b4bc16d419eaf5b14da

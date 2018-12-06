# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:59:12 2018

@author: ldkea_000
"""

<<<<<<< HEAD
        print('---=== Generation', i, '===---')
        if type(render_gens)==int and (i % render_gens) == 0:
            print(simulate(population[0], environment, render=True, max_steps=max_steps, fps=fps))
        if render_gens=='BW_render':
            population_best_worst(population, environment, simulation_reps, i, fps=fps)
        if render_gens=='BW':
            population_best_worst(population, environment, simulation_reps, i, render=False)
        if render_gens=='change':
            population_change(population, environment, i)
        population = run_generation(
=======

<<<<<<< HEAD
print('Building environment...', flush=True)
env = gym.make('MsPacman-ram-v0')
print('Generating population...', flush=True)
population = [GeneticPerceptronActor(env.observation_space, env.action_space) for _ in range(5)]
print('Evolving...', flush=True)
evolve(population, env, generations=50, simulation_reps=1, max_steps=5000, render_gens='BW', allow_parallel=False,fps=100, p_mutation=.01)
=======

                    

'''
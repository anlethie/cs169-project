import gym
from testing import test_actor_class
from actor import GeneticPerceptronActor as GPA
from actor import GeneticNNActor as GNNA

if __name__ == '__main__':
    env = gym.make('Breakout-ram-v0')
    test_actor_class(GNNA, env,
        savefile='Breakout_NN_4_pop_50_pm_15.txt',
        population_size=50,
        actor_args={
                'hidden_layers': [4]
            },
        evolve_args={
                'generations': 1000,
                'simulation_reps':2,
                'max_steps':10000,
                'p_mutation': 0.15,
                'render_gens': None,
                'savenum': 1,
                'allow_parallel':True
            },
        render_args={
                'fps': 20,
                'max_steps':5000
            }
        )

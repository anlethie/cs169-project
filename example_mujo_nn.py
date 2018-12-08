import gym
from testing import test_actor_class
from actor import ModifiedGeneticNNActor as MGNNA

if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    test_actor_class(MGNNA, env,
        savefile='HalfCheetah-v2-GNNAM.txt',
        population_size=200,
        actor_args={
                'hidden_layers': [6,6,6,6,6,6]
            },
        evolve_args={
                'generations': 500,
                'simulation_reps': 3,
                'max_steps': 5000,
                'p_mutation': 0.05,
                'render_gens': 10,
                'savenum': 1,
                'allow_parallel':True
            },
        render_args={
                'fps': 60,
                'max_steps':5000
            }
        )
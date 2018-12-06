import gym
from testing import test_actor_class
from actor import GeneticNNActor as GPA

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    test_actor_class(GPA, env,
        savefile='CartPole.txt',
        actor_args={'hidden_layers':[5]
            },
        evolve_args={
                'generations': 1000,
                'simulation_reps':1,
                'max_steps':1000,
                'p_mutation': 0.10,
                'render_gens': 1,
                'savenum': 1
            },
        render_args={
                'fps': 30,
                'max_steps':5000
            }
        )
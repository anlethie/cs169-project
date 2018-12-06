import gym
from testing import test_actor_class
from actor import GeneticPerceptronActor as GPA

# actor_class, env, population_size=100,
# savefile='test_actor_class.txt', actor_args = {},
# evolve_args = {}, render_args = {}):

if __name__ == '__main__':
    env = gym.make('MsPacman-ram-v0')
    test_actor_class(GPA, env,
        savefile='CartPole.txt',
        actor_args={
            },
        evolve_args={
                'generations': 1000,
                'simulation_reps':5,
                'max_steps':1000,
                'p_mutation': 0.03,
                'render_gens': 5,
                'savenum': 3,
            },
        render_args={
                'fps': 30,
                'max_steps':5000
            }
        )
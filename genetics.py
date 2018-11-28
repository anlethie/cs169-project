import numpy as np
import random as rnd

from execution import simulate

# Having some trouble getting parallel computations to work. For now, disabled.
ALLOW_PARALLEL = False

try:
    assert ALLOW_PARALLEL,'Parallel computation manually disabled.'
    import gym
    from actor import Actor
    from joblib import Parallel, delayed
    # Actually execute a simple test of the parallel system, to make sure it works
    N_JOBS = 8
    # PARALLEL_TEST = Parallel(n_jobs=N_JOBS)(delayed(lambda x: x**0.5)(i) for i in range(100))
    env = gym.make('CartPole-v1')
    PARALLEL_TEST = Parallel(n_jobs=N_JOBS)(delayed(lambda a: simulate(a, env))(a)
        for a in [Actor(env.observation_space, env.action_space) for _ in range(100)])
    USE_PARALLEL  = True
    print('Parallel computation enabled.')
except Exception as e:
    USE_PARALLEL = False
    print('Warning! Parallel computation disabled.')
    print(e)

try:
    from multiprocessing import cpu_count
    N_JOBS = cpu_count()
    print('Detected', N_JOBS, 'CPUs.')
except Exception as e:
    print('Warning! Could not detect number of CPUs. Assuming', N_JOBS, '.')

# A "Genome" is just a Numpy array that's defined to have floating-point data between 0 and 1
# Standardizing what a genome is for our purposes lets us manipulate them in the abstract, 
#   so these functions can be applied to any problem environment.

def mutate(genome, p=1.0):
    """Takes a genome and randomly mutates it.
The probability of mutating each element is p."""
    new_genome = genome.copy()
    for i in range(genome.size):
        if np.random.rand() < p:
            new_genome[i] = np.random.rand()
    return new_genome

def crossover(genome1, genome2):
    """Takes two genomes and performs a cross-over operation at a random point."""
    assert genome1.size == genome2.size
    i = np.random.randint(1, genome1.size-1) # Don't allow 0-length cross-overs
    new_genome1 = genome1.copy()
    new_genome2 = genome2.copy()
    new_genome1[i:] = genome2[i:].copy()
    new_genome2[:i] = genome1[:i].copy()
    return new_genome1,new_genome2


def run_generation(
        population, environment,
        p_mutation=0.01,
        simulation_reps=100,
        max_steps=1000
        ):
    """Executes one full generation, starting with the given population.
Returns the new population.
p_mutation - the chance that a particular value in an offspring genome is changed
simulation_reps - the number of times to execute each actor in the environment, to account for random variation in initial environmental conditions
max_steps - the maximum number of simulation steps for each run
"""
    # TODO: write this function. Sketch:
    # x simulate each actor in population against environment some number of times, average scores
    #   Use those averaged scores to generate next population's genomes, copied from previous population according to roulette wheel or exponential distribution selection
    #     Alternatively, take the selection strategy as a functional parameter, to be tuned later
    #   Use mutate and cross-over on the resulting next generation
    #   Build GeneticActors from those new genomes

    N = len(population)

    def run_actor(actor):
        """Returns the actor's average score and genome."""
        score = np.mean([
            simulate(actor, environment, max_steps=max_steps, render=False)
            for _ in range(simulation_reps)
        ])
        return (score,actor.get_genome())


    population = list(population) # demand that this is a list, not an iterable or something weird.
    if USE_PARALLEL:
        # When available, use parallel evaluation to speed up the evaluation of the population
        scored_genomes = Parallel(n_jobs=N_JOBS)(delayed(run_actor)(actor) for actor in population)
    else:
        # Otherwise, not a big deal, just do things sequentially
        scored_genomes = [run_actor(actor) for actor in population]

    # Taken directly from textbook
    flo = min(s for s,_ in scored_genomes)
    fhi = max(s for s,_ in scored_genomes)
    C = 0.1 * flo + 1.1 * fhi
    D = max(1, fhi + C)
    # Perform the scaling:
    scored_genomes = [( ((score + C) / D) , genome ) for score,genome in scored_genomes]
    total_score = sum(s for s,_ in scored_genomes)
    # Re-write the scores as cumulative scores:
    cscored_genomes = [scored_genomes[0]]
    for score,genome in scored_genomes[1:]:
        cscored_genomes.append((cscored_genomes[-1][0] + score, genome))

    def roulette_sample():
        r = np.random.rand() * total_score
        # Find the first cumulative score that exceeds the random number
        return next(genome for cscore,genome in cscored_genomes if cscore >= r)

    # perform roulette-wheel sampling:
    mating_pool = [roulette_sample() for _ in range(N)]

    def gen_children():
        parents = rnd.sample(mating_pool, 2)
        return crossover(*parents)

    # Perform cross-over N//2 times, since each produces 2 offspring
    # Simultaneously execute mutations on each child
    offspring_genomes = [mutate(x, p_mutation) for _ in range(N // 2) for x in gen_children()]

    # need some actor from the original pool, just to generate the new actors
    actor = population[0]

    return [actor.from_genome(genome) for genome in offspring_genomes]


def evolve(
        initial_population,
        environment,
        generations=100,
        p_mutation=0.01,
        simulation_reps=100,
        max_steps=1000,
        render_gens=10
        ):
    """Runs selection and simulation on initial_population for specified number of generations.
Returns the final generation.
Renders a random individual from the population every render_gens generations.
p_mutation - the chance that a particular value in an offspring genome is changed
simulation_reps - the number of times to execute each actor in the environment, to account for random variation in initial environmental conditions
max_steps - the maximum number of simulation steps for each run
"""
    population = initial_population
    for i in range(generations):
        if (i % render_gens) == 0:
            print('---=== Generation', i, '===---')
            simulate(population[0], environment, render=True)

        population = run_generation(
                population, environment,
                p_mutation=p_mutation,
                simulation_reps=simulation_reps,
                max_steps=max_steps
            )

    return population
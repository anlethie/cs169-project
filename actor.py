from functools import reduce
import numpy as np

def product(l):
    return reduce(lambda x,y: x*y, l)

class Actor:
    """Encapsulates how to respond to observations in some environment."""
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space      = action_space

    def react_to(self, observation):
        """Returns an action in response to the observation."""
        # Without specializing, this is just a random actor
        return self._action_space.sample()


class GeneticActor(Actor):
    def get_genome(self):
        """Returns the genomic representation of self."""
        raise NotImplemented()

    def from_genome(self, genome):
        """Generates a new GeneticActor for the same environment as this actor, based on the given genome."""
        return GeneticActor()



class PerceptronActor(Actor):
    def __init__(self, observation_space, action_space):
        super(PerceptronActor, self).__init__(observation_space, action_space)
        self._n_obs = product(observation_space.shape)
        self._n_act = action_space.n
        self._perceptron_matrix = np.ones((self._n_act, self._n_obs)) / (self._n_act * self._n_obs)

    def react_to(self, observation):
        outputs = self._perceptron_matrix.dot(np.reshape(observation, self._n_obs))
        i = 0
        for j,x in enumerate(outputs):
            if x > outputs[i]:
                i = j
        return i


class GeneticPerceptronActor(PerceptronActor, GeneticActor):
    def get_genome(self):
        return np.reshape(self._perceptron_matrix, self._n_obs * self._n_act)

    def from_genome(self, genome):
        pa = GeneticPerceptronActor(self._observation_space, self._action_space)
        pa._perceptron_matrix = np.reshape(genome, self._perceptron_matrix.shape)
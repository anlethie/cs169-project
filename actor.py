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
        """Returns the genomic representation of self. A genome is a numpy array of floats."""
        return np.array([])

    def from_genome(self, genome):
        """Generates a new GeneticActor for the same environment as this actor, based on the given genome."""
        return GeneticActor(self._observation_space, self._action_space)



class PerceptronActor(Actor):
    def __init__(self, observation_space, action_space):
        super(PerceptronActor, self).__init__(observation_space, action_space)
        self._n_obs = product(observation_space.shape)
        self._n_act = action_space.n
        self._perceptron_matrix = np.random.random((self._n_act, self._n_obs))

    def react_to(self, observation):
        outputs = self._perceptron_matrix.dot(np.reshape(observation, self._n_obs))
        i = 0
        for j,x in enumerate(outputs):
            if x > outputs[i]:
                i = j
        return i

class NeuralNetActor(Actor):
    """Actor that uses a neural network to react to observations."""
    def __init__(self, observation_space, action_space, hidden_layers=[]):
        """hidden_layers is a list of numbers, each number the number of nodes on a hidden layer, in order"""
        super(NeuralNetActor, self).__init__(observation_space, action_space)
        self._n_obs = product(observation_space.shape)
        self._n_act = action_space.n
        self._layers = []
        for in_size,out_size in zip([self._n_obs] + hidden_layers, hidden_layers + [self._n_act]):
            self._layers.append(np.random.random((out_size, in_size)))
        self._threshold_fn = lambda X: 1. / (1. + np.exp(-X))

    def react_to(self, observation):
        current_vector = np.reshape(observation, self._n_obs)
        for layer in self._layers:
            current_vector = layer.dot(current_vector)
            current_vector = self._threshold_fn(current_vector)
        i = 0
        for j,x in enumerate(current_vector):
            if x > outputs[i]:
                i = j
        return i



class GeneticPerceptronActor(PerceptronActor, GeneticActor):
    def get_genome(self):
        return (np.reshape(self._perceptron_matrix.copy(), self._n_obs * self._n_act) + 1) / 2

    def from_genome(self, genome):
        pa = GeneticPerceptronActor(self._observation_space, self._action_space)
        pa._perceptron_matrix = (2 * np.reshape(genome.copy(), self._perceptron_matrix.shape)) - 1
        return pa

class GeneticNNActor(NeuralNetActor, GeneticActor):
    def get_genome(self):
        genome = []
        for layer in self._layers:
            genome.append(np.reshape(layer.copy(), product(layer.shape)))
        genome = (genome + 1.)/2.
        return genome

    def from_genome(self, genome):
        genome = (genome.copy() * 2.) - 1.
        nna = GeneticNNActor(self._observation_space, self._action_space)
        nna._layers = []
        start = 0
        for layer in self._layers:
            layer_size = product(layer.shape)
            new_layer = np.reshape(genome[start:start+layer_size].copy(), layer.shape)
            nna._layers.append(new_layer)
            start += layer_size
        return nna

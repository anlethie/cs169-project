class Actor:
    """Encapsulates how to respond to observations in some environment."""
    def __init__(self, observation_space, action_space):
        self._action_space = action_space
        self._total_reward = 0

    def react_to(self, observation):
        """Returns an action in response to the observation."""
        # With specializing, this is just a random actor
        return self._action_space.sample()
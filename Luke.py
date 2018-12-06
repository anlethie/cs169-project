import numpy as np
def react_to(environment, action_space, genome, observation):
    actions=0
    for action in range(len(20)):
        if environment.action_space.contains(action)==True:
            actions=action
        else:
            break
    reaction=genome[-1]
    for i in range(len(genome)-1):
        reaction+=genome[i]*observation[i]
    reaction=1/(1+np.exp(-reaction))
        
        
        
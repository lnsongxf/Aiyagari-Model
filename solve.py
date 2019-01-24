# Households Problem
# State variables s_t =(a_t,z_t), where a_t is the asset in period t, and z_t is the shock 
# Choice variable a_{t+1}
# R is a matrix where R[s,a] is the reward at state s under action a
# T is a matrix where T[s,a,s'] is the probability of transitioning to state s' when current state is s and current action is a 

import numpy as np
from numba import jit

class Aiyagari:
    def __init__(self,
                 
    

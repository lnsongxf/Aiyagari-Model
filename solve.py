# Households Problem
# State variables s_t =(a_t,z_t), where a_t is the asset in period t, and z_t is the shock 
# Choice variable a_{t+1}
# R is a matrix where R[s,a] is the reward at state s under action a
# T is a matrix where T[s,a,s'] is the probability of transitioning to state s' when current state is s and current action is a 

import numpy as np
from quantecon.markov import DiscreteDP
from numba import jit
import matplotlib.pyplot as plt
import time

st_time = time.time()

class Aiyagari:
    def __init__(self,
                 r = 0.01,
                 w = 1,
                 beta = 0.96,
                 A = 1,
                 N = 1,
                 alpha = 0.33,
                 delta = 0.05,                
                 a_min = 1e-6,
                 a_max = 20,
                 a_size = 100,
                 Pi = [[0.9, 0.1], [0.1,0.9]],
                 z_val = [0.5, 1.5]):

        self.r, self.w, self.beta = r, w, beta
        self.A, self.N, self.alpha, self.delta = A, N, alpha, delta
        self.a_min, self.a_max, self.a_size = a_min, a_max, a_size
        self.Pi = np.asarray(Pi)
        self.z_val = np.asarray(z_val)
        self.z_size = len(z_val)

        self.a_val = np.linspace(a_min, a_max, a_size)
        self.n = a_size * self.z_size

        # build matrix R[s,a]
        self.R = np.full((self.n, a_size),-np.inf)
        build_R(self.R, self.a_size, self.z_size, self.a_val, self.z_val, self.r, self.w)

        # build matrix T[s,a,s']
        self.T = np.zeros((self.n, a_size, self.n))
        build_T(self.T, self.a_size,self.z_size, self.Pi)

        dp = DiscreteDP(self.R, self.T, self.beta)
        solution = dp.solve(method='policy_iteration')

        # Optimal actions across the set of a indices with z fixed in each row
        a_star = np.zeros((self.z_size, a_size))
        for s_i in range(self.n):
            a_i = s_i // self.z_size
            z_i = s_i % self.z_size
            a_star[z_i, a_i] = self.a_val[solution.sigma[s_i]]

        # Create grid for r
        nr = 20
        r_val = np.linspace(0.005, 0.04, nr)

        # Create supply for capital
        k_val = np.empty(nr)
        for i, r in enumerate(r_val):
            self.r = r
            w = A*(1-alpha)*(A*alpha/(r+delta))**(alpha/(1-alpha))
            self.w = w
            build_R(self.R, self.a_size, self.z_size, self.a_val, self.z_val, self.r, self.w)
            aiyagari_dp = DiscreteDP(self.R, self.T, self.beta)

            # Compute the optimal policy
            aiyagari_policy = aiyagari_dp.solve(method='policy_iteration')
    
            # Compute the stationary distribution
            stationary_probs = aiyagari_policy.mc.stationary_distributions[0]

            # Compute the marginal distribution for assets
            asset_probs = np.zeros(a_size)
            for a_i in range(a_size):
                for z_i in range(self.z_size):
                    asset_probs[a_i] += stationary_probs[a_i*self.z_size+z_i]
            # Compute capital supply k
            k_supply = np.sum(asset_probs*self.a_val)
            k_val[i] = k_supply

        # Capital demand is given by FOC of firm problem
        rd = np.empty(nr)
        for i in range(nr):
            rd[i] = A*alpha*(N/k_val[i])**(1-alpha)-delta
             
        # Plot capital supply and demand
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_val, r_val, lw=2, alpha=0.6, label='supply of capital')       
        ax.plot(k_val, rd, lw=2, alpha=0.6, label='demand for capital')
        ax.grid()
        ax.set_xlabel('capital')
        ax.set_ylabel('interest rate')
        ax.legend(loc='upper right')
        #plt.show()
        

        ed_time = time.time()
        print("Running time:")
        print(ed_time - st_time)       

       
@jit(nopython=True)
def build_R(R, a_size, z_size, a_val, z_val, r, w):
    n = a_size*z_size
    for s_i in range(n):
        a_i  = s_i // z_size
        z_i = s_i % z_size
        a = a_val[a_i]
        z = z_val[z_i]
        for a_i_new in range(a_size):
            a_new = a_val[a_i_new]
            c = w * z + (1+r) * a - a_new
            if c > 0:
                R[s_i, a_i_new] = np.log(c)

@jit(nopython=True)
def build_T(T, a_size, z_size,Pi):
    n = a_size*z_size
    for s_i in range(n):
        z_i = s_i % z_size
        for a_i in range(a_size):
            for z_i_next in range(z_size):
                T[s_i, a_i, a_i*z_size+z_i_next]=Pi[z_i,z_i_next]
                
    



if __name__=="__main__":
    ae = Aiyagari()
    print('Done')               
                 
                 
    

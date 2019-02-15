"""
Softmax Algorithm
"""
import math
import numpy as np

class Softmax():
    def __init__(self, temperature, n_arms):
        self.temperature = temperature
        self.n_arms = n_arms
        return
    
    
    def initialize(self, size_batch):
        self.size_batch = size_batch
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        return
    
    def select_arm(self, X_hat):
        z = sum([math.exp(v/self.temperature) for v in self.values])
        probs = [math.exp(v/self.temperature)/z for v in self.values]
        selected_arms = list(np.random.multinomial(self.size_batch, probs, 1)[0])  
        return selected_arms
    
    
    def update(self, pulls, successes):
        ''' 
        Update the count and the values vector.
        For count: add + 1 to the arm chosen at time t
        For values: Observe the reward obtained at time t, and average it with
        the previously estimated value
        N.B: as n increases, the weight of the new observation decreases.
        '''
        self.counts = np.array(pulls) + self.counts 
        for arm in range(self.n_arms):
            if pulls[arm] > 0:
                n = self.counts[arm]
                value = self.values[arm]
                new_value = ((n-1)*value + successes[arm])/float(n)
                self.values[arm] = new_value
        return
    
    def name(self):
        return 'Softmax, tau=' + str(self.temperature)

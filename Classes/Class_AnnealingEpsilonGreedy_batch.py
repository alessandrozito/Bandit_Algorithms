"""
This file contains the Annealing Epsilon Greedy class of bandit algorithms.
The algo works a s follows: (1) explore with a probability epsilon
                            (2) exploit with a probability 1 - epsilon
N.B: Epsilon decreases over time. This favors exploration in earlier times,
and allows for exploitation in later periods. The idea is that, once we have 
understood what arm is the best, there is no need to keep on exploring. 
"""
import random
import numpy as np
import math

class AnnealingEpsilonGreedy():
    """ Annealing Epsilon-Greedy algorithm """
    
    def __init__(self, n_arms):
        """
        Initialize the class
        """
        self.n_arms = n_arms
        return
  
    def initialize(self, size_batch):
        """
        Initialize a vector of counts to count the number of times the arm is chosen 
        within the horizon, and a vector of values that contains the estimated
        value of every arm based on the rewards obtained
        """
        self.size_batch = size_batch
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        return
    
    def ind_max(self, x):
        ''' An easy function to select the arm according to its position'''
        if type(x) != 'list':
            x = list(x)
        m = max(x)
        return x.index(m) 
    
    def select_arm(self, X_hat):
        """
        If a random number is greater than epsilon, pull the best arm
        Otherwise, test another arm at random. Epsilon is a decreasing 
        function of time
        """ 
        selected_arms = [0 for arm in range(self.n_arms)]
        obs = self.size_batch
        
        while obs > 0:
            t = sum(self.counts) + 1 
            epsilon = 1/math.log(t + 0.000001)
            if random.random() > epsilon:
                chosen_arm =  self.ind_max(self.values)
            else:
                chosen_arm =  random.randrange(len(self.values))
            selected_arms[chosen_arm] = selected_arms[chosen_arm] + 1
            obs = obs - 1
            
        return selected_arms

      
    # Finally, update You need this to see which arm is considered best
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
        return 'Annealing epsilon greedy'









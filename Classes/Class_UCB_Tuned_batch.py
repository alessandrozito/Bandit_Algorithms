'''
This file contains the UCB1 class of bandit algorithms.
The enviromnet is striclty deterministic. At each time t, the algorithm computes 
a confidence bound for each arm and selects the arm that corrisponds to the highest
bound. This favors exploration over exploitation: When an arm is not pulled for a long
time, its bound will generally increase and make it more likeble.
The idea is to exploit the best arm as more as possible, but still monitor the
situation of other arms. This is particularly efficient when, for uncertain reasons
the best arm chases to be the best and its payoff drops.

This algorithm is further tuned to acount for the variance in the arms.
'''
import math
import numpy as np

class UCB_Tuned():
    """ UCB1 algorithm """
    
    def __init__(self, n_arms):
        """
        Initialize the class
        Note: delta = ['1/t', '1/t^2', else == number]
        """
        self.n_arms = n_arms
        return
    
    def initialize(self, size_batch):
        """
        Initialize a vector of counts to count the number of times the arm is chosen 
        within the horizon, and a vector of values that contains the estimated
        value of every arm based on the rewards obtained
        """
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.values_squared =np.zeros(self.n_arms)
        self.size_batch = size_batch
        return
    
    def select_arm(self, X_hat):
        '''
        For each arm, compute the confidence bound and select the one with the 
        highest bound. If the arm has never been selected, pull it. This allows the UCB1
        to get info on every arm before it starts to exploit
        '''
        selected_arms = [0 for arm in range(self.n_arms)]
        
        for arm in range(self.n_arms):
            if self.counts[arm]==0:
                selected_arms[arm] = self.size_batch
                return selected_arms
            
        ucb_values = [0.0 for arm in range(self.n_arms)]
        total_counts = sum(self.counts)
        
        '''
        Set the confidence level. Accont for the variance as well
        '''         
        for arm in range(self.n_arms):
            variance = self.values_squared[arm] - (self.values[arm]**2) + math.sqrt(2*math.log(total_counts) / self.counts[arm])
            bonus = math.sqrt((math.log(total_counts) / float(self.counts[arm]))*min(0.25, variance))
            ucb_values[arm] = self.values[arm] + bonus
        
        selected_arms[self.ind_max(ucb_values)] = self.size_batch
        return selected_arms
    
    def ind_max(self, x):
        if type(x) != 'list':
            x = list(x)
        m = max(x)
        return x.index(m) 
 
    
    
    def update(self, pulls, successes):
        ''' 
        Update the count and the values vector.
        For count: add + 1 to the arm chosen at time t
        For values: Observe the reward obtained at time t, and average it with
        the previously estimated value
        N.B: as n increases, the weight of the new observation decreases.
        '''
        chosen_arm = pulls.index(self.size_batch)
        reward = successes[chosen_arm]
        self.counts[chosen_arm] = self.counts[chosen_arm] + self.size_batch
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n-1)*value + reward)/float(n)
        self.values[chosen_arm] = new_value
        
        value_sq = self.values_squared[chosen_arm]
        new_value_sq = ((n-1)*value_sq + reward**2)/float(n)
        self.values_squared[chosen_arm] = new_value_sq
        return

    def name(self):
        return 'UCB Tuned'




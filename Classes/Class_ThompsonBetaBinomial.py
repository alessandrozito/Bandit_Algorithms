# -*- coding: utf-8 -*-
"""
Beta-binomial RPM
"""

import numpy as np
from scipy.stats import beta

''' Thompson Gibbs '''
class ThompsonBetaBinomial():

    def __init__(self, a, b, size_wat):
        # The prior over the probability of success is given by a beta distribution
        # N.B: a and b must be vectors, with lenght given by the number of arms
        self.a = a
        self.b = b
        self.size_wat = size_wat
        return

    def initialize(self, size_batch):
        self.n = np.zeros(len(self.a))
        self.y = np.zeros(len(self.a))
        self.size_batch = size_batch
        return

    def ind_max(self, x):
        if type(x) != 'list':
            x = list(x)
        m = max(x)
        return x.index(m)

    def select_arm(self, X_hat):
        I_a_theta = np.array([0 for arm in range(len(X_hat.columns))])

        # Draw #size_wat theta for each arm, and count the number of times each one is
        # the best
        thetaMult = np.transpose(list(map(lambda a_arm,b_arm, n_arm, y_arm:
                beta.rvs(a_arm+y_arm, b_arm+n_arm-y_arm, size=self.size_wat),
                self.a , self.b,  self.n,  self.y)))

        for g in range(self.size_wat):
            theta_mt = thetaMult[g]
            highest_arm = self.ind_max(theta_mt)
            I_a_theta[highest_arm] = I_a_theta[highest_arm] + 1

        w_at = I_a_theta * (1/self.size_wat)
        selected_arms = list(np.random.multinomial(self.size_batch, w_at, 1)[0])
        return selected_arms

    def update(self, pulls, successes):
        ''' Update the matrix X and the response vector y by one dimension'''
        self.n = self.n + pulls
        self.y = self.y + successes
        return

    def name(self):
        return 'Beta-Binomial Thompson Sampling, particles = '+ str(self.size_wat)

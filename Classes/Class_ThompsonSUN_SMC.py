# -*- coding: utf-8 -*-
"""
Fractional Factorial Thompson Sampling via unified skew-normal and Sequential
Monte Carlo
"""

''' Packages '''
import pandas as pd
import numpy as np
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal as mtv_norm


import sys
''' Intoduce the Unified Skewed Normal sampler and call it'''
sys.path.insert(0, r'C:\Users\zito1\Desktop\MBA_catatrofale\bandit_algorithms\Classes')
sys.path.insert(0,r'/Users/antonio/Desktop/bandit_algorithms/Classes')
sys.path.insert(0, '/home/alezitoart/bandit_algorithms/Classes' )
from SUN import SUN
SUN = SUN()


class ThompsonSUN_SMC():

    def __init__ (self, mu, Sigma, size_wat, t_cutoff):
        ''' Initialize the class with the priors, and the matrix of Successes
        and failures'''
        self.mu = mu
        self.Sigma = Sigma
        self.size_wat = size_wat
        self.t_cutoff = t_cutoff
        return

    def initialize(self, size_batch):
        self.X =  pd.DataFrame()
        self.y = np.array([])
        self.theta = np.array([])
        self.size_batch = size_batch
        return

    def ind_max(self, x):
        ''' An easy function to select the arm according to its position'''
        if type(x) != 'list':
            x = list(x)
        m = max(x)
        return x.index(m)

    def select_arm(self, X_hat):

        #N.B X_hat ahs shape p x n, it is already transposed
        I_a_theta = np.array([0 for arm in range(len(X_hat.columns))])

        n =  self.X.shape[0]
        if n == 0:
            ''' Cold start: if nothing has been observed, sample from the prior
            (multivariate normal)'''
            self.theta = mtv_norm.rvs(mean = self.mu, cov =  self.Sigma,\
             size= self.size_wat)

        elif n <  self.t_cutoff:
            ''' Draw one theta from the posterior distribution given X and y'''
            self.theta = np.array(SUN.posterior_draw(self.mu, self.Sigma,\
             self.y, self.X, size=self.size_wat))
        else:
            # Keep the last results from beta and from theta
            y_batch = self.y[n- self.size_batch:]
            X_batch = self.X.loc[n- self.size_batch:]
            # Compute the resampling weights
            absolute_weights = np.array(list(map(lambda beta:    \
             np.prod(norm.cdf((2*y_batch -1)*np.matmul(X_batch, beta))), \
             self.theta)))
            importance_weights = absolute_weights/sum(absolute_weights)
            ''' Resample from the previous particles '''
            self.theta = random.choices(self.theta, weights=importance_weights, \
            k=self.size_wat)

        thetaMult = np.matmul(self.theta, X_hat)

        for g in range(self.size_wat):
            theta_mt = thetaMult[g]
            highest_arm = self.ind_max(theta_mt)
            I_a_theta[highest_arm] = I_a_theta[highest_arm] + 1

        w_at = I_a_theta * (1/self.size_wat)
        selected_arms = list(np.random.multinomial(self.size_batch, w_at, 1)[0])

        return selected_arms


    def update(self, X_selected, responses):
        ''' Update the matrix X and the response vector y by one dimension'''
        self.X = self.X.append(X_selected, ignore_index = True)
        self.y = np.append(self.y, responses)
        return

    def name(self, horizon):
        if self.t_cutoff > horizon:
            return 'SUN Thompson Sampling'
        else:
            return 'SUN Thompson Sampling, SMC at n=' + str(self.t_cutoff)

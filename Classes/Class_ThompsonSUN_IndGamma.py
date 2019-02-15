# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:02:45 2018

@author: zito1
"""

''' Packages '''
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mtv_norm


import sys
''' Intoduce the Unified Skewed Normal sampler and call it'''
sys.path.insert(0, r'C:\Users\zito1\Desktop\MBA_catatrofale\bandit_algorithms\Classes')
sys.path.insert(0,r'/Users/antonio/Desktop/bandit_algorithms/Classes')
sys.path.insert(0, '/home/alezitoart/bandit_algorithms/Classes' )
from SUN_IndGamma import SUN_IndGamma
SUN = SUN_IndGamma()


class ThompsonSUN_IndGamma():
    
    def __init__ (self, mu, Sigma, size_wat):
        ''' Initialize the class with the priors, and the matrix of Successes and failures'''
        self.mu = mu
        self.Sigma = Sigma
        self.size_wat = size_wat
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
            ''' Cold start: if nothing has been observed, sample from the prior (multivariate normal)'''
            self.theta = mtv_norm.rvs(mean = self.mu, cov =  self.Sigma, size= self.size_wat)
            
        else:    
            ''' Draw one theta from the posterior distribution given X and y'''
            self.theta = np.array(SUN.posterior_draw(self.mu, self.Sigma, self.y, self.X, size=self.size_wat))
            print(self.theta)
            
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

    def name(self):
        return 'SUN Thompson Sampling, Independent Gamma matrix'








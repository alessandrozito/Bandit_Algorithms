# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:18:11 2018

@author: zito1
"""

import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal as mtv_norm

#%% Import packages to run to R scripts
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
from rpy2.robjects.packages import importr

#%% Install and import the TruncatedNormal package
packnames = ['bayesm']
from rpy2.robjects.vectors import StrVector
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
TruncatedNormal = importr('bayesm')
rbprobitGibbs = ro.r['rbprobitGibbs']

#%%
from rpy2.robjects import numpy2ri
numpy2ri.activate()
''' Thompson Gibbs '''
class ThompsonGibbs():
    
    def __init__(self, mu, Sigma, burn_in, size_gibbs):
        # The prior over theta in multivariate normal with mean vector b and location matrix Sigma
        self.mu = mu
        self.Sigma = Sigma
        self.burn_in = burn_in
        self.size_gibbs = size_gibbs
        return
    
    def initialize(self, size_batch):
        self.X =  pd.DataFrame()
        self.y = np.array([])
        self.mu_r = ro.FloatVector(self.mu)
        self.Sigma_r =  ro.r['matrix'](np.linalg.inv(np.array(self.Sigma)), ncol=len(self.Sigma))
        self.size_batch = size_batch
        return 
    
    def ind_max(self, x):
        if type(x) != 'list':
            x = list(x)
        m = max(x)
        return x.index(m) 
    
    def draw_z(self, y, mean_tr):
        if y == 1.0:
            z = truncnorm.rvs(0, float('Inf'), mean_tr, 1)
        elif y == 0.0:
            z = truncnorm.rvs(float('-Inf'),0, mean_tr, 1)
        return z
    
    def select_arm(self, X_hat):
        
        n = self.X.shape[0]
        I_a_theta = np.array([0 for arm in range(len(X_hat.columns))])
        
        if n == 0:
            # COLD START: At time zero, select one arm at random using a draw from 
            #             the normally distributed prior
            theta = mtv_norm.rvs(mean= self.mu, cov = self.Sigma, size = self.size_gibbs) 
            thetaMult = np.matmul(theta, X_hat)
        
        else: 
            # Gibbs Draws: Sample form approximated posterior distribution given the batch update 
            y_r = ro.FloatVector(self.y)
            X_r = ro.r['matrix'](np.array(self.X), ncol=len(self.mu))
            
            # Call the ALbChib R Function , and draw the posterior samples
            Data_GIBBS = ro.ListVector({'y':y_r, 'X': X_r})
            Prior_GIBBS =  ro.ListVector({'betabar':self.mu_r, 'A': self.Sigma_r})
            MCMC_GIBBS = ro.ListVector({'R': self.burn_in + self.size_gibbs, 'keep': 1, 'nprint':0 })
            GIBBS_Samples = np.array(rbprobitGibbs(Data=Data_GIBBS, Prior=Prior_GIBBS, Mcmc=MCMC_GIBBS).rx2('betadraw'))[self.burn_in:]
            # Update the best arm vector
            thetaMult = np.matmul(GIBBS_Samples, X_hat)
            
        for g in range(self.size_gibbs):
            theta_mt = thetaMult[g]
            highest_arm = self.ind_max(theta_mt) 
            I_a_theta[highest_arm] = I_a_theta[highest_arm] + 1
        w_at = I_a_theta * (1/self.size_gibbs)
        selected_arms = list(np.random.multinomial(self.size_batch, w_at, 1)[0])  
        return selected_arms
        

    def update(self, X_selected, responses):
        ''' Update the matrix X and the response vector y by one dimension'''
        self.X = self.X.append(X_selected, ignore_index = True)
        self.y = np.append(self.y, responses)
        return 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:04:25 2018

@author: antonio
"""
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm
from scipy.stats import multivariate_normal as mtv_norm
import math 

class ThompsonGibbs_Scottversion():
    
    def __init__(self, mu, Sigma, burn_in, size_gibbs, X_arms):
        # The prior over theta in multivariate normal with mean vector b and location matrix Sigma
        self.mu = mu
        self.Sigma = Sigma
        self.burn_in = burn_in
        self.size_gibbs = size_gibbs
        self.X_arms = X_arms
        return
    
    def initialize(self, size_batch):
        self.inv_Sigma = np.linalg.inv(self.Sigma)
        self.Omega_inv = self.inv_Sigma
        self.Omega = np.linalg.inv(self.Omega_inv)
        self.Sigma_inv_mu = np.matmul(self.inv_Sigma, self.mu)
        self.arm_n = np.array([0 for col in range(self.X_arms.shape[0])])   # Vector that counts the number of times the arm has been palyed
        self.arm_y = np.array([0 for col in range(self.X_arms.shape[0])]) # count the number of successes so far (our y_at)
        self.size_batch = size_batch
        return
    
    ''' Define the asymptotic distribution of the censored data '''
    def lambda_alpha(self, alpha):
        return norm.pdf(alpha)/(1-norm.cdf(alpha))
    def delta_alpha(self, alpha):
        return norm.pdf(alpha)/norm.cdf(alpha)
    
    def ind_max(self, x):
        if type(x) != 'list':
            x = list(x)
        m = max(x)
        return x.index(m)

    def draw_z(self, sign, arm, mu_a):
        
        ''' Simulate from the normal distribution truncated to the left'''
        if sign == '+':
            size = int(self.arm_y[arm])
            '''
            If the number of successes so far is lower than 50, simulate from 
            A simple normal distribution censored to the left at 0
            z_a+ is the sum of y_a draws from this distribution
            '''
            if size <= 50:
                a = 0 - mu_a
                b = float('Inf')
                z_plus = sum(truncnorm.rvs(a, b, mu_a, 1, size))

            else:
                '''
                If the number of successes is higher than 50, then use the asymptotic 
                distribution as in Appendix A.2 in Scott(2010)
                '''
                mean = size*(mu_a + self.lambda_alpha(-mu_a))
                variance = size*(1 - self.lambda_alpha(-mu_a)*(self.lambda_alpha(-mu_a) + mu_a))
                z_plus = norm.rvs(loc = mean, scale = math.sqrt(variance))
            return z_plus
        
       
        # Simulate from the normal distribution truncated to the right 
        elif sign == '-' :
            '''
            If the number of failures so far is lower than 50, simulate from 
            a simple normal distribution censored to the right at 0
            z_a- is the sum of n_a - y_a draws from this distribution
            '''
            size = int(self.arm_n[arm] - self.arm_y[arm])
            if size <= 50:
                a  = float('-Inf')
                b = 0 - mu_a
                z_minus = sum(truncnorm.rvs(a,b, mu_a, 1, size))   
            # If the number of successes is higher than 50, then use the asymptotic 
            # distribution as in Appendix A.2 in Scott(2010)
            else:
                mean = size*(mu_a- self.delta_alpha(-mu_a))
                variance = size*(1 + mu_a*self.delta_alpha(-mu_a) - (self.delta_alpha(- mu_a) ** 2))
                z_minus = norm.rvs(loc = mean, scale = math.sqrt(variance))
            return z_minus


    def select_arm(self, X_hat):
        n = sum(self.arm_n)
        I_a_theta = np.array([0 for arm in range(len(X_hat.columns))])
        
        if n == 0:
            # COLD START: At time zero, select one arm at random using a draw from 
            #             the normally distributed prior
            theta = mtv_norm.rvs(mean= self.mu, cov = self.Sigma, size = self.size_gibbs) 
            thetaMult = np.matmul(theta, X_hat)
        
        else: 
            thetas = pd.DataFrame()
            # Gibbs Draws: Sample form approximated posterior distribution given the batch update 
            theta = mtv_norm.rvs(mean= self.mu, cov = self.Sigma, size = 1) 
            arms = list(X_hat.columns)
            for g in range(self.size_gibbs + self.burn_in):
                thetas = thetas.append(pd.Series(theta), ignore_index = True)
                arm_means = np.matmul(theta,X_hat)
                # Step1: Draw zeta
                z = np.array(list(map(lambda arm, mu: self.draw_z('+', arm, mu) + self.draw_z('-', arm, mu), arms, arm_means)))
                # Step2: Draw theta
                XTz = (z[:, np.newaxis] * np.transpose(X_hat)).sum(axis = 0)
                theta_tilda = np.matmul(self.Omega, XTz + self.Sigma_inv_mu)
                theta = mtv_norm.rvs(mean=theta_tilda, cov = self.Omega)
                
            thetaMult = np.matmul(thetas[self.burn_in:], X_hat)
            
        for g in range(self.size_gibbs):
            theta_mt = thetaMult[g]
            highest_arm = self.ind_max(theta_mt) 
            I_a_theta[highest_arm] = I_a_theta[highest_arm] + 1
            
        w_at = I_a_theta * (1/self.size_gibbs)
        selected_arms = list(np.random.multinomial(self.size_batch, w_at, 1)[0])  
        return selected_arms
               
    
    def update(self, pull, successes):
        self.arm_n = self.arm_n + pull
        self.arm_y = self.arm_y + successes
        self.Omega_inv =  self.Omega_inv + np.matmul(np.transpose(self.X_arms),np.matmul(np.diag(pull), self.X_arms))
        self.Omega = np.linalg.inv(self.Omega_inv)
        return 
    
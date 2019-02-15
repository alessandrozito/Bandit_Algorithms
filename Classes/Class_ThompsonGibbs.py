# -*- coding: utf-8 -*-
"""
Fractional Factorial Thompson Sampling via ALbert and Chib (1993)
"""

import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal as mtv_norm

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
        self.Sigma_inv_mu = np.matmul(np.linalg.inv(self.Sigma), self.mu)
        self.Omega_inv = np.linalg.inv(self.Sigma)
        self.Omega = np.linalg.inv(self.Omega_inv)
        self.size_batch = size_batch
        return

    def ind_max(self, x):
        if type(x) != 'list':
            x = list(x)
        m = max(x)
        return x.index(m)

    def draw_z(self, y,  mean_tr):
        if y == 1:
            a = 0 - mean_tr
            b = np.inf
        else:
            a = -np.inf
            b = 0 - mean_tr
        z = truncnorm.rvs(a = a, b= b,loc = mean_tr)
        return float(z)

    def select_arm(self, X_hat):

        n = self.X.shape[0]
        I_a_theta = np.array([0 for arm in range(len(X_hat.columns))])

        if n == 0:
            # COLD START: At time zero, select one arm at random using a draw from
            #             the normally distributed prior
            theta = mtv_norm.rvs(mean= self.mu, cov = self.Sigma, size = self.size_gibbs)
            thetaMult = np.matmul(theta, X_hat)

        else:

            theta = self.mu
            XT = np.transpose(self.X)
            #theta_Gibbs = pd.DataFrame()
            thetas = pd.DataFrame([])

            for g in range(self.burn_in + self.size_gibbs):
                #################################################
                # Step 1: Sample z form the truncated normal given theta
                #################################################
                means = np.matmul(self.X, np.transpose(theta))
                z = list(map(lambda y_r, m_r: self.draw_z(y_r, m_r), self.y, means))
                #################################################
                # Step 2: Sample theta given the vector zeta
                #################################################
                XTz = np.matmul(XT, z)
                theta_tilda = np.matmul(self.Omega, XTz + self.Sigma_inv_mu)
                theta = mtv_norm.rvs(mean=theta_tilda, cov = self.Omega)
                thetas = thetas.append(pd.Series(theta), ignore_index=True)

            thetaMult = np.matmul(thetas[self.burn_in:], X_hat)

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
        self.Omega_inv =  self.Omega_inv + np.matmul(np.transpose(X_selected), X_selected)
        self.Omega = np.linalg.inv(self.Omega_inv)
        return

    def name(self):
        return 'MCMC Thompson Sampling'

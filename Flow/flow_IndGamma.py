#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:51:47 2019

@author: antonio
"""

""" Launch the simulation"""

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from Class_ThompsonSUN_IndGamma import ThompsonSUN_IndGamma
from Test_final import test_algorithm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import math 


def join_list(list):
    list  = [x for x in list for x in x]
    return list

def generate_and_test(algo_dict, arms_dict, horizon, size_batch):
        
    ''' Generate the bandit environment '''
    X_arms = pd.DataFrame(arms_dict)
    X_hat = np.transpose(X_arms)

    ''' Generate a new vector of success probabilities '''
    X_arms = pd.DataFrame(arms_dict)
    alpha = list(np.random.normal(-1.645, math.sqrt(0.1) , 1))
    params = list([np.random.normal(0, math.sqrt(0.25)) for col in range(len(X_arms.columns)-1)])
    theta = alpha + params
    X_arms['success_probability'] = norm.cdf(np.matmul(X_arms, theta))
    X_arms['arm'] = X_arms.index
    #a4_dims = (12, 10)
    #fig, ax = plt.subplots(figsize=a4_dims)
    #sns.distplot(X_arms['success_probability'], bins = 15, rug= True, kde=False, ax=ax).set(xlabel='Success Probability', ylabel='Frequency')
    
    ''' Run The algorithm with the associated arms'''
    results = list(map(lambda algo_name: test_algorithm(algo_dict[algo_name], X_arms, X_hat, horizon, size_batch), 
                       list(algo_dict.keys())))
    results_all = pd.concat(x for x in results)
    return results_all
    
def parallel_simulations(algo_dict, arms_dict, horizon, size_batch, jobs, sims):
    simulations = Parallel(n_jobs = jobs)(delayed(generate_and_test)(algo_dict, arms_dict, horizon, size_batch) for n in range(sims))
    results = pd.concat(x for x in simulations)
    return results


def main():
    n_arms = 30
        
    arms_dict = { 
                 'constant' : [1.0 for arm in range(n_arms)],
                 # Featrure 1
                 'x_1' : [1.0 for arm in range(int(n_arms/2))] + [0.0 for arm in range(int(n_arms/2))],
                 # Featrure 2
                 'x_2_1' : [1.0 for arm in range(int(n_arms/(2*3)))]  + 
                              [0.0 for arm in range(int(n_arms/(2*3)))] +
                              [0.0 for arm in range(int(n_arms/(2*3)))] +
                              [1.0 for arm in range(int(n_arms/(2*3)))] +
                              [0.0 for arm in range(int(n_arms/(2*3)))] +
                              [0.0 for arm in range(int(n_arms/(2*3)))] ,
                 'x_2_2' : [0.0 for arm in range(int(n_arms/(2*3)))] +
                              [1.0 for arm in range(int(n_arms/(2*3)))] +
                              [0.0 for arm in range(int(n_arms/(2*3)))] +
                              [0.0 for arm in range(int(n_arms/(2*3)))] +
                              [1.0 for arm in range(int(n_arms/(2*3)))] +
                              [0.0 for arm in range(int(n_arms/(2*3)))],
                # Featrure 3
                'x3_1' : join_list([[1, 0, 0, 0, 0] for arm in range(int(n_arms/5))]),
                'x3_2' : join_list([[0, 1, 0, 0, 0] for arm in range(int(n_arms/5))]),
                'x3_3' : join_list([[0, 0, 1, 0, 0] for arm in range(int(n_arms/5))]),
                'x3_4' : join_list([[0, 0, 0, 1, 0] for arm in range(int(n_arms/5))])
                }


    X_arms = pd.DataFrame(arms_dict)
    # Simulation horizon
    n_sims =40
    horizon = 400
    jobs = 1
    size_batch = 1
    # prior parameters
    mu = np.zeros(len(X_arms.columns))
    Sigma = np.identity(len(X_arms.columns))
    # Algorithm dictionary:
    algo_dict = {
        'algo_SUN' : ThompsonSUN_IndGamma(mu, Sigma, 5000)
        }
    #run the simulation results
    simulations_results = parallel_simulations(algo_dict, arms_dict, horizon, size_batch, jobs, n_sims)
    
    return simulations_results

# Run the algorithm
simulations_results = main()
flierprops = dict(markerfacecolor='0.55', markersize=2,
              linestyle='none')
sns.set(font_scale = 1.3)
sns.set_style('whitegrid')
g = sns.FacetGrid(simulations_results, col="algorithm", col_wrap=2, height = 7)
g = g.map(sns.lineplot, 'times', 'regrets')
ticks = list(range(0, 400 + 1, 100))
xlabs = [str(i) for i in ticks]
plt.xticks(ticks, xlabs)
plt.show()



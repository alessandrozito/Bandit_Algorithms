#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:43:58 2019

@author: antonio
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:58:10 2018

@author: zito1
"""

""" Launch the simulation"""

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from Class_ThompsonBetaBinomial import ThompsonBetaBinomial
from Test_final import test_algorithm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
 
s = 0


def join_list(list):
    list  = [x for x in list for x in x]
    return list

def generate_and_test(algo_dict, arms_dict, horizon, size_batch, s):
    s = s + 1
    print(s)
    ''' Generate the bandit environment '''
    X_arms = pd.DataFrame(arms_dict)
    X_hat = np.transpose(X_arms)

    ''' Generate a new vector of success probabilities '''
    X_arms = pd.DataFrame(arms_dict)
    alpha = [0.1]
    params = [0.4, 0.3, 0.2, 0.1]
    theta = alpha + params
    #theta = np.random.beta(1,6, 100)
    X_arms['success_probability'] = np.matmul(X_arms, theta)
    X_arms['arm'] = X_arms.index
    ''' Run The algorithm with the associated arms'''
    results = list(map(lambda algo_name: test_algorithm(algo_dict[algo_name], X_arms, X_hat, horizon, size_batch), 
                       list(algo_dict.keys())))
    results_all = pd.concat(x for x in results)
    return results_all
    
def parallel_simulations(algo_dict, arms_dict, horizon, size_batch, jobs, sims):
    simulations = Parallel(n_jobs = jobs)(delayed(generate_and_test)(algo_dict, arms_dict, horizon, size_batch, s) for n in range(sims))
    results = pd.concat(x for x in simulations)
    return results


def main():
    n_arms = 5
    
    
    arms_dict = { 'constant': [1.0 for arm in range(n_arms)],
                  'x_1': [1,0,0,0,0],
                  'x_2': [0,1,0,0,0],
                  'x_3': [0,0,1,0,0],
                  'x_4': [0,0,0,1,0]
                }
    
    #arms_dict = {}
    #for arm in range(n_arms):
    #    arms_dict['x_' + str(arm + 1)] = [0.0 for arm in range(n_arms)]
    #    arms_dict['x_' + str(arm + 1)][arm] = 1.0
        
    
    # Simulation horizon
    n_sims =100
    horizon = 1000
    jobs = 1
    size_batch = 1
    # prior parameters
    a = np.ones(n_arms)
    b = np.ones(n_arms)
    # Algorithm dictionary:
    algo_dict = {
        'algo_BetaBin_TS' : ThompsonBetaBinomial(a,b, 1),    
        'algo_BetaBin_RPM' : ThompsonBetaBinomial(a,b, 1000)
        }
    #run the simulation results
    simulations_results = parallel_simulations(algo_dict, arms_dict, horizon, size_batch, jobs, n_sims)
    
    return simulations_results

# Run the algorithm
simulations_results = main()
#simulations_results.to_csv('SUN_final_comparison.csv', index=False)
flierprops = dict(markerfacecolor='0.55', markersize=2,
              linestyle='none')
sns.set(font_scale = 1.3)
sns.set_style('whitegrid')
g = sns.FacetGrid(simulations_results, col="algorithm", col_wrap=2, height = 7)
g.set_axis_labels("Times", "Per-Period Regret")
g = g.map(sns.lineplot, 'times', 'regrets')
ticks = list(range(0, 1000 + 1, 200))
xlabs = [str(i) for i in ticks]
plt.xticks(ticks, xlabs)
plt.show()
g.savefig('BIN_comparison_many_arms_3.png')


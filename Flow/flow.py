# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:58:10 2018

@author: zito1
"""

""" Launch the simulation"""

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from Class_ThompsonGibbs import ThompsonGibbs
from Class_ThompsonSUN_SMC import ThompsonSUN_SMC
from Class_ThompsonBetaBinomial import ThompsonBetaBinomial
from Test_scott import test_algorithm
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
    params = list([np.random.normal(0, math.sqrt(0.1)) for col in range(len(X_arms.columns)-1)])
    theta = alpha + params
    X_arms['success_probability'] = norm.cdf(np.matmul(X_arms, theta))
    X_arms['arm'] = X_arms.index
    
    ''' Run The algorithm with the associated arms'''
    results = list(map(lambda algo_name: test_algorithm(algo_dict[algo_name], X_arms, X_hat, horizon, size_batch), 
                       ['algo_BetaBin', 'algo_Gibbs',  'algo_SUN']))
    results_all = pd.concat(x for x in results)
    return results_all
    
def parallel_simulations(algo_dict, arms_dict, horizon, size_batch, jobs, sims):
    simulations = Parallel(n_jobs = jobs)(delayed(generate_and_test)(algo_dict, arms_dict, horizon, size_batch) for n in range(sims))
    results = pd.concat(x for x in simulations)
    return results


def main():
    n_arms = 120
    arms_dict = { 
                 'constant' : [1 for arm in range(n_arms)],
                 # Featrure 1 - 2 levels
                 'x1' : [1 for arm in range(int(n_arms/2))] + [0 for arm in range(int(n_arms/2))],
                 # Featrure 2 - 3 levels
                 'x2_1' : [1 for arm in range(int(n_arms/(2*3)))]  + 
                              [0 for arm in range(int(n_arms/(2*3)))] +
                              [0 for arm in range(int(n_arms/(2*3)))] +
                              [1 for arm in range(int(n_arms/(2*3)))] +
                              [0 for arm in range(int(n_arms/(2*3)))] +
                              [0 for arm in range(int(n_arms/(2*3)))] ,
                 'x2_2' : [0 for arm in range(int(n_arms/(2*3)))] +
                              [1 for arm in range(int(n_arms/(2*3)))] +
                              [0 for arm in range(int(n_arms/(2*3)))] +
                              [0 for arm in range(int(n_arms/(2*3)))] +
                              [1 for arm in range(int(n_arms/(2*3)))] +
                              [0 for arm in range(int(n_arms/(2*3)))],
                              
                # Featrure 3 - 4 levels
                 'x3_1' : join_list(join_list([([1 for n in range(5)], [0 for n in range(15)]) for x in range(6)])),
                 'x3_2' : join_list(join_list([([0 for n in range(5)], [1 for n in range(5)], [0 for n in range(10)]) for x in range(6)])),
                
                 'x3_3' : join_list(join_list([([0 for n in range(10)], [1 for n in range(5)], [0 for n in range(5)]) for x in range(6)])),
                              
                # Featrure 4 - 5 levels
                'x4_1' : join_list([[1, 0, 0, 0, 0] for arm in range(int(n_arms/5))]),
                'x4_2' : join_list([[0, 1, 0, 0, 0] for arm in range(int(n_arms/5))]),
                'x4_3' : join_list([[0, 0, 1, 0, 0] for arm in range(int(n_arms/5))]),
                'x4_4' : join_list([[0, 0, 0, 1, 0] for arm in range(int(n_arms/5))])
                
                }

    X_arms = pd.DataFrame(arms_dict)
    # Simulation horizon
    n_sims =100
    horizon = 100
    jobs = 40
    size_batch = 100
    day_cutoff = 6
    # prior parameters
    mu = np.zeros(len(X_arms.columns))
    Sigma = np.identity(len(X_arms.columns))
    a = np.ones(n_arms)
    b = np.ones(n_arms)
    # Algorithm dictionary:
    algo_dict = {
        'algo_BetaBin' : ThompsonBetaBinomial(a,b, 1000),
        'algo_Gibbs' : ThompsonGibbs(mu, Sigma, 0, 1000),
        'algo_SUN' : ThompsonSUN_SMC(mu, Sigma, 1000, day_cutoff*size_batch)
        }
    #run the simulation results
    simulations_results = parallel_simulations(algo_dict, arms_dict, horizon, size_batch, jobs, n_sims)
    
    return simulations_results

# Run the algorithm
simulations_results = main()

simulations_results.to_csv('Scott_AlbChib_SUN_BinTS_Comparison_v3.csv', index=False)

flierprops = dict(markerfacecolor='0.75', markersize=3,
              linestyle='none')
g = sns.FacetGrid(simulations_results, col="algorithm", col_wrap=1, height = 4, aspect = 2.5)
g = g.map(sns.boxplot, 'times', 'regrets', flierprops = flierprops)
ticks = list(range(0, 101, 20))
xlabs = [str(i) for i in ticks]
plt.xticks(ticks, xlabs)


g.savefig('Scott_AlbChib_SUN_BinTS_Comparison_v3.png')
"""
sns.lineplot(x='times', y='completion_time', hue='algorithm', data=simulations_results)
"""



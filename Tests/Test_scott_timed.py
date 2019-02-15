# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:44:35 2018

@author: zito1
"""

import numpy as np
import pandas as pd
import time
import random

#%% Useful Class
class Arm_response():
    def __init__(self, p):
        self.p = p
        
    def singledraw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0
        
    def sumdraw(self, size): 
        draw = sum([self.singledraw() for d in range(size)])
        return draw
    
#%%
        
def test_algorithm(algo, X_arms, X_hat, horizon, size_batch):
    
    ''' Build the results to store'''
    simulation_dict = {
        'times' : [0 for col in range(horizon)],
        'best_arm_counter' : [0 for col in range(horizon)],
        'best_prob_counter' : [0.0 for col in range(horizon)],
        'rewards' : [0.0 for col in range(horizon)],
        'cumulative_rewards' : [0.0 for col in range(horizon)],
        'regrets' : [0.0 for col in range(horizon)],
        'cumulative_regrets' : [0.0 for col in range(horizon)],
        'completion_time' : [0.0 for col in range(horizon)],
        'algorithm': [str(algo.__class__.__name__)  for col in range(horizon)],
        'time_cutoff': [str(algo.return_time_cutoff()) + ' seconds' for col in range(horizon)]
        }
    
    best_arm = int(X_arms[X_arms['success_probability'] == max(X_arms['success_probability'])]['arm'])
    best_prob = max(X_arms['success_probability'])
    

    # Initialize the quantities 
    algo.initialize(size_batch)
    arms_responses = list(map(lambda p: Arm_response(p), X_arms['success_probability']))
    probs = list(X_arms['success_probability'])
    # Start the cycle
    for t in range(horizon):
        print(t)
        day_time = time.time()
        t = t + 1
        index = t - 1
        
        # Draw for the posterior and select the correpsonding arm
        chosen_arms = algo.select_arm(X_hat)
        if algo.__class__.__name__ != 'ThompsonSUN_SMC':
            successes = np.array(list(map(lambda arm, size: arms_responses[arm].sumdraw(size), list(X_hat.columns), chosen_arms)))
            reward = sum(successes)
            regret = sum(list(map(lambda x: chosen_arms[x] * (best_prob - probs[x]), list(X_hat.columns))))
            algo.update(chosen_arms, successes)  
            
        else: 
            X_day = pd.DataFrame([])
            y_day = np.array([])
            regret = 0
            
            for arm in range(len(X_hat.columns)):
                pulls = chosen_arms[arm]
                while pulls > 0:
                    chosen_arm_features = np.array(X_hat[arm])
                    response = arms_responses[arm].singledraw()
                    X_day = X_day.append(pd.Series(chosen_arm_features), ignore_index = True)
                    y_day = np.append(y_day, response)
                    chosen_prob = float(X_arms[X_arms['arm'] == arm]['success_probability'])
                    regret = regret + best_prob - chosen_prob
                    pulls = pulls - 1 
             
            reward = sum(y_day)
            # Update the matrix and the corresponding y
            algo.update(X_day, y_day)    
        
        # Update the quantities in the dictionary
        simulation_dict['times'][index] = t
        simulation_dict['best_arm_counter'][index] = best_arm
        simulation_dict['best_prob_counter'][index] = best_prob
        simulation_dict['regrets'][index] = regret
        simulation_dict['rewards'][index] = reward
        
        if t == 1:
            simulation_dict['cumulative_rewards'][index] = reward
            simulation_dict['cumulative_regrets'][index] = regret
        else:
            simulation_dict['cumulative_rewards'][index] = simulation_dict['cumulative_rewards'][index-1] + reward
            simulation_dict['cumulative_regrets'][index] = simulation_dict['cumulative_regrets'][index-1] + regret
        simulation_dict['completion_time'][index] = time.time() - day_time           
        
    return pd.DataFrame(simulation_dict)

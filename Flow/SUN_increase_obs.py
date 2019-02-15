import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r'/Users/antonio/Desktop/bandit_algorithms/Classes')
sys.path.insert(0, '/home/alezitoart/bandit_algorithms/Classes' )
from SUN import SUN
SUN = SUN()

sns.set(font_scale = 1.5)
sns.set_style('whitegrid')

"""""""""""
Figure 4: 1000 draws form a Bea distribution
"""""""""""
n = 1000
df_beta = pd.DataFrame()
df_beta_1 = pd.DataFrame({ 'prob_a1': np.random.beta(20,30,n),
                         'prob_a2': np.random.beta(2,1,n)})
df_beta_1['draws'] = 'arm 1: Beta(20,30) - arm 2: Beta(2,1)'
df_beta_2 = pd.DataFrame({ 'prob_a1': np.random.beta(20,30,n),
                         'prob_a2': np.random.beta(20,10,n)})
df_beta_2['draws'] = 'arm 1: Beta(20,30) - arm 2: Beta(20,10)'
df_beta_3 = pd.DataFrame({ 'prob_a1': np.random.beta(20,30,n),
                         'prob_a2': np.random.beta(60,30,n)})
df_beta_3['draws'] = '1: Beta(20,30) - arm 2: Beta(60,30)'
df_beta = df_beta.append(df_beta_1, ignore_index=True)
df_beta = df_beta.append(df_beta_2, ignore_index=True)
#df_beta = df_beta.append(df_beta_3, ignore_index=True)
df_beta.loc[ df_beta['prob_a1'] > df_beta['prob_a2'], 'best'] = 'arm a_1 is best'
df_beta.loc[ df_beta['prob_a1'] < df_beta['prob_a2'], 'best'] = 'arm a_2 is best'
g = sns.FacetGrid(df_beta, hue= 'best', col="draws", height = 7, 
                  hue_kws={"color": ["blue","red" ]}, col_wrap=3)
g = g.map(plt.scatter, 'prob_a1', 'prob_a2', alpha = 0.5, edgecolor="white")
g.set_axis_labels("Estimated Success Probability of arm 1", \
                  "Estimated Success probability of arm 2");
g.set(xticks=list(np.linspace(0,1, 11)), yticks=list(np.linspace(0,1, 11)))
plt.show()
g.savefig('Beta_observations_3.png')

"""""""""""
Figure 9: 1000 draws form a SUN distribution
"""""""""""
alpha_true = -0.5
beta_true = 0.5

n_10 = 10
n_100 = 100
dict_10 = {'constant': np.ones(n_10),
         'x_a': np.random.binomial(1, 0.5, n_10)}
dict_100 = {'constant': np.ones(n_100),
         'x_a': np.random.binomial(1, 0.5, n_100)}
df_10 = pd.DataFrame(dict_10)
df_100 = pd.DataFrame(dict_100)
df_10['y'] = np.random.binomial(1, norm.cdf(alpha_true * df_10['constant']
                                     + beta_true * df_10['x_a']), n_10)

df_100['y'] = np.random.binomial(1, norm.cdf(alpha_true * df_100['constant']
                                     + beta_true * df_100['x_a']), n_100)

mu = np.zeros(2)
Sigma = np.identity(2)

draws_10 = SUN.posterior_draw(mu,Sigma, df_10['y'], \
                              df_10[['constant', 'x_a']], 1000)
draws_100 = SUN.posterior_draw(mu,Sigma, df_100['y'], \
                               df_100[['constant', 'x_a']], 1000)
df_draws = pd.DataFrame()
df_draws_10 = pd.DataFrame()
df_draws_100 = pd.DataFrame()
df_draws_10['alpha_hat'] = np.transpose(draws_10)[0] 
df_draws_10['beta_hat'] = np.transpose(draws_10)[1] 
df_draws_10['size'] = '10 observations'
df_draws_100['alpha_hat'] = np.transpose(draws_100)[0] 
df_draws_100['beta_hat'] = np.transpose(draws_100)[1] 
df_draws_100['size'] = '100 observations'

df_draws = df_draws.append(df_draws_10, ignore_index = True)
df_draws = df_draws.append(df_draws_100, ignore_index = True)
df_draws['prob_a1'] = norm.cdf(df_draws['alpha_hat'] + df_draws['beta_hat'])
df_draws['prob_a2'] = norm.cdf(df_draws['alpha_hat'])

df_draws.loc[ df_draws['prob_a1'] > df_draws['prob_a2'], 'best'] =\
         'arm a_1 is best'
df_draws.loc[ df_draws['prob_a1'] < df_draws['prob_a2'], 'best'] =\
         'arm a_2 is best'

g = sns.FacetGrid(df_draws, hue= 'best', col="size", height = 7, 
                  hue_kws={"color": ["blue", 'red' ]}, col_wrap=3)
g = g.map(plt.scatter, 'prob_a1', 'prob_a2', alpha = 0.5, edgecolor="white")
g.set_axis_labels("Estimated Success Probability of arm 1", \
                  "Estimated Success probability of arm 2");
g.set(xticks=list(np.linspace(0,1, 11)), yticks=list(np.linspace(0,1, 11)))
g.savefig('SUN_observations_3.png')







#Reproduce the picturs for the conclusion of the thesis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('SUN_final_comparison.csv')

algos = list(df.groupby('algorithm').groups.keys())
flierprops = dict(markerfacecolor='0.55', markersize=3,
              linestyle='none')
sns.set(font_scale = 2)
sns.set_style('whitegrid')
dims = (24, 12)
for algo in algos:
    df_algo = df[df['algorithm']==algo]
    fig, ax = plt.subplots(figsize=dims)
    ax.set(xlabel='Per-period regret', ylabel='Times')
    if algo=='MCMCM Thompson Sampling':
        algo = 'Gibbs Thompson Sampling'
    elif algo== 'SUN Thompson Sampling, SMC at n=0':
        algo = 'SUN Thompson Sampling, SMC at t=0'
    elif algo== 'SUN Thompson Sampling, SMC at n=200':
        algo = 'SUN Thompson Sampling, SMC at t=200'
    sns.boxplot(x ='times', y = 'regrets',data =df_algo,  \
                      flierprops=flierprops, ax=ax).set_title(algo)
    ticks = list(range(0, 400 + 1, 100))
    xlabs = [str(i) for i in ticks]
    plt.xticks(ticks, xlabs)
    plt.savefig('Final_comparison_'+ algo +'.png')
    plt.show()
    
ffts_list =  ['MCMCM Thompson Sampling','SUN Thompson Sampling',
              'SUN Thompson Sampling, SMC at n=0', 'SUN Thompson Sampling, SMC at n=200']
df_ffts = df[df['algorithm'].isin(ffts_list)]
#rename the variables
df_ffts.loc[df_ffts['algorithm'] == 'MCMCM Thompson Sampling', 'algorithm'] = 'Gibbs FFTS'
df_ffts.loc[df_ffts['algorithm'] == 'SUN Thompson Sampling', 'algorithm'] = 'SUN FFTS'
df_ffts.loc[df_ffts['algorithm'] == 'SUN Thompson Sampling, SMC at n=0', 'algorithm'] = 'SUN FFTS, SMC at t=0'
df_ffts.loc[df_ffts['algorithm'] == 'SUN Thompson Sampling, SMC at n=200', 'algorithm'] = 'SUN FFTS, SMC at t=200'

#%%
# Create a figure instance, and the two subplots
fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
bins=25
sns.distplot(df_ffts[(df_ffts['times']==400)&\
                     (df_ffts['algorithm']=='Gibbs FFTS')]['cumulative_regrets'], hist=False, ax=ax1, bins=bins)
sns.distplot(df_ffts[(df_ffts['times']==400)&\
                     (df_ffts['algorithm']=='SUN FFTS')]['cumulative_regrets'], hist=False,ax=ax1, bins=bins)
sns.distplot(df_ffts[(df_ffts['times']==400)&\
                     (df_ffts['algorithm']=='SUN FFTS, SMC at t=0')]['cumulative_regrets'], hist=False, ax=ax1, bins=bins)
sns.distplot(df_ffts[(df_ffts['times']==400)&\
                     (df_ffts['algorithm']=='SUN FFTS, SMC at t=200')]['cumulative_regrets'], hist=False,  ax=ax1, bins=bins)

ax1.set_xlabel('Cumulative regret at T=400')
ax1.set_ylabel('Density')

# Tell the factorplot to plot on ax2 with the ax argument
# Also store the FacetGrid in 'g'
sns.lineplot(x='times',y='completion_time',hue='algorithm', data=df_ffts, ci=None, ax=ax2)
ax2.set_xlabel('Times')
ax2.set_ylabel('Seconds')
# Close the FacetGrid figure which we don't need (g.fig)
ax2.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2), ncol=5)
plt.show()

"""
Figure 8: Shapes of the unified skew-normal density 
"""
import pandas as pd
import numpy as np
import seaborn as sns
import sys
sys.path.insert(0,r'/Users/antonio/Desktop/bandit_algorithms/Classes')
sys.path.insert(0, '/home/alezitoart/bandit_algorithms/Classes' )
from SUN import SUN
SUN = SUN()

"""Compute the Delta qunatity """
def fnDelta(x, y):
    delta = (2*y - 1)* x * ((x**2 + 1)**(-0.5))
    return delta

""" Plot the graphs """
line = []
for x in range(10):
    line = line + (list(np.linspace(-3, 3 , num = 1000, endpoint=True)))
x = []
x = x + [-3.0 for col in range(int(len(line)/(2*5)))] 
x = x + [-1.5 for col in range(int(len(line)/(2*5)))]
x = x + [0.0 for col in range(int(len(line)/(2*5)))] 
x = x + [1.5 for col in range(int(len(line)/(2*5)))] 
x = x + [3.0 for col in range(int(len(line)/(2*5)))] 

graph = {'Real Line' : line,
         'r' : [1 for col in range(int(len(line)/2))] +\
         [0 for col in range(int(len(line)/2))],
         'x' :  x + x}

df = pd.DataFrame(graph)
df['Delta'] = fnDelta(df['x'], df['r'])
df['Density'] = SUN.pdf(df['Real Line'],0,1, df['Delta'], 0,1)
sns.set(font_scale = 1.3)
sns.set_style('whitegrid')
sns.set_palette("mako")
sns.relplot(x = 'Real Line', y = 'Density', col='r', style='x', data=df, \
            kind="line")


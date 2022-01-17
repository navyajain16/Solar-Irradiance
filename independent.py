# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:33:14 2022

@author: navya
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

data1 = pd.read_csv('SolarData2019.csv')
data = data1.iloc[:,:-3]
label_encoder = LabelEncoder()
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
f, ax = plt.subplots(figsize=(8, 11)) 
plt.title("Correlation among different variables",size=11)
heatmap = sns.heatmap(corr, 
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4, 
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1, 
                      vmax = 1,
                      annot = False,
                      annot_kws = {'size': 12})
#add the column names as labels
ax.set_yticklabels(corr.columns, rotation = 0)
ax.set_xticklabels(corr.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.savefig('correlation.pdf',bbox_inches='tight')
#sns.heatmap(corr, cmap="coolwarm")

#Correlation with output variable
cor_target = abs(corr["GHI"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.75:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
data = data[selected_columns]

target = data1.columns[10]
print(target)
X = data1.drop(target,axis=1)
corr1= X.corr()
columns1 = np.full((corr1.shape[0],), True, dtype=bool)
selected_columns = X.columns[columns1]
#import statsmodels.formula.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(data1.iloc[:,1:].values, data1.iloc[:,0].values, SL, selected_columns)
result = pd.DataFrame()
print(selected_columns)
#result['GHI'] = data.iloc[:,0]
#from statistics import mean
#mean(result['GHI'])

data = pd.DataFrame(data = data_modeled, columns = selected_columns)
print(selected_columns)



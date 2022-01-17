# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:51:56 2022

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

df = pd.read_csv('SolarData2019.csv')
X = df.drop("GHI",1)   #Feature Matrix
y = df["GHI"]          #Target Variable
df.head()

#Adding constant column of ones, mandatory for sm.OLS model
#X_1 = sm.add_constant(X)
#Fitting sm.OLS model
model = sm.OLS(y,X).fit()
for attributeIndex in range (0, 17):
    print(model.pvalues[attributeIndex]) 

    #pp = model.pvalues[attributeIndex]
p1= model.pvalues
p1

pp = [0.7007187293770498, 3.391897948096692e-16,
0.7811855727091972,
0.3367490559180324,
1.8225265872633511e-127,
5.904225464713618e-81,
3.0717613226277194e-227,
0.6240653724973565,
0.0,
0.0,
2.6369636746628516e-33,
0.04293622232370438,
1.6797697239756997e-41,
1.5096039294568884e-28,
0.19703192426601177,
4.491950181532512e-21,
0.24015704262979284]


# who v/s fare barplot
cols = list(X.columns)
dataa = pd.DataFrame(list(zip(cols,pp )),
               columns =['Feature', 'p-value'])
ax = sns.barplot(x = dataa['Feature'],
            y = dataa['p-value'],
            data = dataa, color = "pink")
ax.axhline(0.05)
plt.title("P-values",size=11)
plt.xlabel('Feature',size=11)
plt.ylabel('P-value',size=11)
plt.rc('xtick',labelsize=11)
plt.rc('ytick',labelsize=11)
plt.tick_params(axis='x', labelsize=11, rotation=75)
plt.tick_params(axis='y', labelsize=11)
plt.savefig('pvalue.pdf',bbox_inches='tight')
# Show the plot
plt.show()

#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
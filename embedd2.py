# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:35:44 2022

@author: navya
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as LGR,Lasso
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.datasets import load_boston


df = pd.read_csv('SolarData2019.csv')
X = df.drop("GHI",1)   #Feature Matrix
y = df["GHI"]    
              # target feature
features = X.columns
features

# Lets apply the Lasso Model -- in Linear Dataset
lasso = Lasso(alpha=0.05)
lasso.fit(X,y)

coeff = lasso.coef_
coeff

df_coeff = pd.DataFrame({"features":features,"coeff":coeff})
df_coeff.sort_values("coeff")

# Use Bar chart to show coeff
df_coeff.set_index('coeff')
# sort in ascending order to better visualization.
df_coeff = df_coeff.sort_values('coeff')
# plot the feature coeff in bars.
plt.figure(figsize=(10,6))
#plt.xticks(rotation=60)
sns.barplot(x="features",y= "coeff", data=df_coeff,color="blue")
plt.title("Feature importance using Lasso model",size=18)
plt.xlabel('Feature',size=18)
plt.ylabel('Coefficients',size=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.tick_params(axis='x', labelsize=18, rotation=75)
plt.tick_params(axis='y', labelsize=18)
plt.savefig('embedd.pdf',bbox_inches='tight')
plt.show()
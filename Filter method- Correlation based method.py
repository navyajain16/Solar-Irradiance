import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm

data = pd.read_csv('C:/solar/SolarData2019.csv') # Reading data
label_encoder = LabelEncoder() #Labelling the festures
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')

# Calculating correlation 
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
f, ax = plt.subplots(figsize=(8, 11))  # Plotting correlation map
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
plt.savefig('correlation.pdf',bbox_inches='tight') #Save fig


#Correlation with output variable
cor_target = abs(corr["GHI"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features # Selected features with GHI




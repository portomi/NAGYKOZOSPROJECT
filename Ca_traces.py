# -*- coding: utf-8 -*-
"""
Created on Mon May 28 00:01:16 2018

@author: tamas_000
"""

import pickle as pkl
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist

### TO DO ###
# input_files = /path/to/inputdir
# output_files = /path/to/outputdir

#%%
### TO DO ###
# input_files = /path/to/inputdir
# output_files = /path/to/outputdir
#%%
#Ca traces to DataFrame
def Ca2DF(Ca_traces, labels):
    
    '''Ca_traces: 3-d numpy array; labels: list of strings
    returns pandas DataFrame, one cell per column'''
    
    dims = map(int, Ca_traces.shape)[:-1]
    Ca_traces_2D=np.zeros(dims) #This is to reduce dimensionality
    for i in range(dims[0]):
        for j in range(dims[1]):
            Ca_traces_2D[i][j] = Ca_traces[i][j][0]
    return(pd.DataFrame(Ca_traces_2D.T, columns=labels))
#%%
def Centroid(poly):
    """Calculates centroid of each individual polygon"""
    poly = pd.DataFrame(poly)
    return poly.mean(axis = 0).tolist()
    
#%%    
"""load data from pickle"""
with open('dFoF_0.pkl', 'rb') as f:
    data = pkl.load(f) #data for Ca signals

# with open('transients_0.pkl', 'rb') as g:
#     threshold_data = pickle.load(g) #data for thresholds ie. noise

with open('rois.pkl', 'rb') as h:
    cell_data = pkl.load(h) #data for IDs of cells

centroids = list() # list for ROI centroids
labels = list() # list for labels of ROIs
#%%
"""Iterate data to calculate each centroid and store it in "centroids" list"""
for i in range(len(cell_data['ROIs']['rois'])):
    #polygons is a 1 element array so [0] after ['polygons'] should stay 0
    centroids.append(Centroid(cell_data['ROIs']['rois'][i]['polygons'][0]))
    labels.append(str(cell_data['ROIs']['rois'][i]['id']))
#%%
"""convert to Dataframe with ROI labels as indexes"""
centroids = pd.DataFrame(centroids)
centroids.insert(loc=0, column='label', value=labels)
centroids.set_index('label')
centroids.columns = ['label', 'x', 'y', 'z']

"""sets distance of 'z' according to the measurement plane distances"""
centroids.loc[centroids['z']==1, 'z'] = 12.5
centroids.loc[centroids['z']==2, 'z'] = 25

"""create distance matrices"""
distances = pdist(centroids.values[:,1:4], metric='euclidean')
indexes = []
for elements in itertools.combinations(labels, 2):
    indexes.append(elements[0] + '__' + elements[1])
distances = pd.DataFrame(distances, index=indexes,
                         columns = ['distance'])   
    
"""Ca traces to DataFrame"""
Ca_traces = np.array(data['ROIs']['traces'])
Ca_traces = Ca2DF(Ca_traces, labels)

#%%
#this is temp. don't run
def calcCorrs(df, groupbycols=None, xcol=None, ycol=None):
    """Calculates crosscorrelation values between two np.array 
    and returns them as pandas dataframe"""
    
    groups = df.groupby(groupbycols)
    
    result = dict()
    result["g1"]= []
    result["g2"]= []
    result["corrValue"] = []
    
    for key1, group1 in groups:
        g1 = group1.sort_values(xcol)
        g1Data = np.array(g1[ycol])
        g1Mask = np.isnan(g1Data)
        for key2, group2 in groups:
            g2 = group2.sort_values(xcol)
            g2Data = np.array(g2[ycol])
            g2Mask = np.isnan(g2Data)
            
            mask = g1Mask | g2Mask
            
            corrval=sp.stats.pearsonr(g1Data[~mask], g2Data[~mask])[0]
            result["g1"].append(key1)
            result["g2"].append(key2)
            result["corrValue"].append(corrval)
            
    return pd.DataFrame(result)
#%%    
def getCellID(data):
    """takes dictionary, returns pandas dataframe""" 
#cell IDs from cell_data
    labels = []
    for x in range(len(cell_data['ROIs']['rois'])):
        labels.append(str(cell_data['ROIs']['rois'][x]['label']))
        return pd.DataFrame(labels)
    
#%%   
#decide confidence interval, i don't think we need this
conf_iv = 0.01
#conf_iv = 0.05
#%%
#Ca traces to DataFrame
Ca_traces = np.array(data['ROIs']['traces'])
Ca_traces = Ca2DF(Ca_traces, labels)
#Ca_traces.to_csv("Ca_signals.csv", encoding="ascii")

#%%
"""Calculating correlations
    where
        PCC is Pearson's Correlation Coefficient
    and
        p-value roughly indicates the probability of an uncorrelated system
        producing datasets that have a Pearson correlation at least as extreme
        as the one computed from these datasets
    """
correlations = {}
columns = Ca_traces.columns.tolist()

for col_a, col_b in itertools.combinations(columns, 2):
    correlations[col_a + '__' + col_b] = pearsonr( \
            Ca_traces.loc[:, col_a], Ca_traces.loc[:, col_b])

result = pd.DataFrame.from_dict(correlations, orient='index')
result.columns = ['PCC', 'p-value']

result = pd.concat([result, distances], axis=1) # puts distances in Ca_traces result DataFrame

result = result.sort_values('PCC')
print(result)
#%%

#plt.pcolormesh(result)
plt.scatter(result['PCC'], result['distance'], c='red', marker='*')
plt.xlabel('Pearsonr correlation')
plt.ylabel('Eucledian distance in um')
plt.show()

#%%
plt.plot(result['PCC'], c='gray')
plt.show()

"""
#calculate correllation between the calculated correlations and data in distance matrix
ssc = StandardScaler()
Correlation = pd.DataFrame(ssc.fit_transform(Correlation))
distance_matrix = pd.DataFrame(ssc.transform(pd.read_csv('Distance_matrix.csv').set_index('Unnamed: 0')))
dist = Correlation.corrwith(distance_matrix)
>>>>>>> Stashed changes

# output_files

pp = PdfPages('corr_dist_mtx.pdf')
fig = plt.figure()
plt.plot(dist)
plt.show()
fig.savefig(pp, format='pdf')
pp.close()
"""

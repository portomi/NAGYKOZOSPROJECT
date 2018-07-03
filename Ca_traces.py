# -*- coding: utf-8 -*-
"""
Created on Mon May 28 00:01:16 2018

@author: tamas_000
"""

import pickle
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

def Ca2DF(Ca_traces, labels):
    """Ca traces to DataFrame"""
    dims = map(int, Ca_traces.shape)[:-1]
    Ca_traces_2D=np.zeros(dims) #This is to reduce dimensionality
    for i in range(dims[0]):
        for j in range(dims[1])
            Ca_traces_2D[i][j] = Ca_traces[i][j][0]
    return(pd.DataFrame(Ca_traces_2D.T, columns=labels))

def Centroid(poly):
    """Calculates centroid of each individual polygon"""
    poly = pd.DataFrame(poly)
    return poly.mean(axis = 0).tolist()
    
    
"""load the pickles"""
with open('dFoF_0.pkl', 'rb') as f:
    data = pickle.load(f) #data for Ca signals
with open('transients_0.pkl', 'rb') as g:
    threshold_data = pickle.load(g) #data for thresholds ie. noise
with open('rois.pkl', 'rb') as h:
    cell_data = pickle.load(h) #data for IDs of cells

centroids = [] # list for ROI centroids
labels = [] # list for labels of ROIs

"""Iterate data to calculate each centroid and store it in "centroids" list"""
for i in range(len(cell_data['ROIs']['rois'])):
    #polygons is a 1 element array so [0] after ['polygons'] should stay 0
    centroids.append(Centroid(cell_data['ROIs']['rois'][i]['polygons'][0]))
    labels.append(str(cell_data['ROIs']['rois'][i]['id']))

"""convert to Dataframe with ROI labels as indexes"""
centroids = pd.DataFrame(centroids)
centroids.insert(loc=0, column='label', value=labels)
centroids.set_index('label')
centroids.columns = ['label', 'x', 'y', 'z']

"""sets distance of 'z' according to the measurement plane distances"""
centroids.loc[centroids['z']==1, 'z'] = 12.5
centroids.loc[centroids['z']==2, 'z'] = 25

"""create distance matrics"""
distances = pdist(centroids.values[:,1:4], metric='euclidean')
indexes = []
for elements in itertools.combinations(labels, 2):
    indexes.append(elements[0] + '__' + elements[1])
distances = pd.DataFrame(distances, index=indexes,
                         columns = ['distance'])   
    
"""Ca traces to DataFrame"""
Ca_traces = np.array(data['ROIs']['traces'])
Ca_traces = Ca2DF(Ca_traces, labels)

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
    correlations[col_a + '__' + col_b] = pearsonr(Ca_traces.loc[:, col_a], Ca_traces.loc[:, col_b])

result = pd.DataFrame.from_dict(correlations, orient='index')
result.columns = ['PCC', 'p-value']

result = pd.concat([result, distances], axis=1) # puts distances in Ca_traces result DataFrame

result = result.sort_values('PCC')
print(result)
#plt.pcolormesh(result)
plt.plot(result['PCC'], result['distance'], c='red', marker='*')
plt.xlabel('Pearsonr correlation')
plt.ylabel('Eucledian distance in um')
plt.show()



# output_files

pp = PdfPages('corr_dist_mtx.pdf')
fig = plt.figure()
plt.plot(dist)
plt.show()
fig.savefig(pp, format='pdf')
pp.close()
"""

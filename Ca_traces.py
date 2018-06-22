# -*- coding: utf-8 -*-
"""
Created on Mon May 28 00:01:16 2018

@author: tamas_000
"""
import this



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

### TO DO ###
# input_files = /path/to/inputdir
# output_files = /path/to/outputdir

#Ca traces to DataFrame
def Ca2DF(Ca_traces, labels):
    dims = map(int, Ca_traces.shape)[:-1]
    Ca_traces_2D=np.zeros(dims) #This is to reduce dimensionality
    for i in range(dims[0]):
        for j in range(dims[1])
            Ca_traces_2D[i][j] = Ca_traces[i][j][0]
    return(pd.DataFrame(Ca_traces_2D.T, columns=labels))
    
#load the pickles
data = pickle.load(open('dFoF_0.pkl', 'rb')) #data for Ca signals
threshold_data = pickle.load(open('transients_0.pkl', 'rb')) #data for thresholds ie. noise
cell_data = pickle.load(open('rois.pkl', 'rb')) #data for IDs of cells

#cell IDs from cell_data
labels = []
for x in range(len(cell_data['ROIs']['rois'])):
    labels.append(str(cell_data['ROIs']['rois'][x]['label']))
    
#decide confidence interval
conf_iv = 0.01
#conf_iv = 0.05

#Ca traces to DataFrame
Ca_traces = np.array(data['ROIs']['traces'])
Ca_traces = Ca2DF(Ca_traces, labels)
#Ca_traces.to_csv("Ca_signals.csv", encoding="ascii")


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

print(result.sort_index())

#Ez mi ez? És miért add 167 sor eredményt?
for col in range(len(Ca_traces.columns)):
   correlations2 = np.correlate(Ca_traces.loc[col], Ca_traces.loc[col+1], mode='full')

"""
#calculate correllation between the calculated correlations and data in distance matrix
ssc = StandardScaler()
Correlation = pd.DataFrame(ssc.fit_transform(Correlation))
distance_matrix = pd.DataFrame(ssc.transform(pd.read_csv('Distance_matrix.csv').set_index('Unnamed: 0')))
dist = Correlation.corrwith(distance_matrix)

# output_files

pp = PdfPages('corr_dist_mtx.pdf')
fig = plt.figure()
plt.plot(dist)
plt.show()
fig.savefig(pp, format='pdf')
pp.close()
"""

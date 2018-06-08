# -*- coding: utf-8 -*-
"""
Created on Mon May 28 00:01:16 2018

@author: tamas_000
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

### TO DO ###
# input_files = /path/to/inputdir
# output_files = /path/to/outputdir

#Ca traces to DataFrame
def Ca2DF(Ca_traces, labels):
    Ca_traces_2D=np.zeros(shape=(84, 3599)) #This is to reduce dimensionality
    for i in range(84):
        for j in range(3599):
            Ca_traces_2D[i][j] = Ca_traces[i][j][0]
    return(pd.DataFrame(Ca_traces_2D.T, columns=labels))
    
def Check_significance(Ca_traces, CI):
    
    """Checks trace's significance.
    Gives back the DataFrame with significant traces only."""

    #treshold calculated from baseline data according to confidence interval
    threshold = threshold_data['ROIs']['transients']['parameters']['thresholds_p'+str(CI)][5]
    for i in range(3599):
        for j in range(84):
            if Ca_traces.iat[i, j]<threshold:
                Ca_traces.iat[i, j] = 0
    return(Ca_traces)
                
                
    
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

#Ca_traces with significant value
Ca_traces = Check_significance(Ca_traces, conf_iv)
#Ca_traces.to_csv("Ca_signals.csv", encoding="ascii")


#calculate correlation between columns
Correlation = Ca_traces.corr()

#calculate correllation betwen the calculated correlations and data in distance matrix
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


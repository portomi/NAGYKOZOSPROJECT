# -*- coding: utf-8 -*-
"""
Created on Sat May 26 20:22:20 2018

@author: tamas_000

ROISs is a dictionary containing an array containing a dictionary
    containing an array of 1 element containing an array.
    So simple and handy as F...k
"""
import pickle
import pandas as pd
from scipy.spatial.distance import pdist, squareform

data = pickle.load(open('rois.pkl', 'rb'))


#Calculates centroid of each individual polygon
def Centroid(poly):
    poly = pd.DataFrame(poly)
    return poly.mean(axis = 0).tolist()
    

#list for ROI centroids
centroids = []
#list for labels of ROIs
labels = []

#Iterate data to calculate each centroid and store it in "centroids" list
for i in range(len(data['ROIs']['rois'])):
    #polygons is a 1 element array so [0] after ['polygons'] should stay 0
    centroids.append(Centroid(data['ROIs']['rois'][i]['polygons'][0]))
    labels.append(str(data['ROIs']['rois'][i]['label']))


#convert to Dataframe with ROI labels as indexes
centroids = pd.DataFrame(centroids)
centroids.insert(loc=0, column='label', value=labels)
centroids.set_index('label')
centroids.columns = ['label', 'x', 'y', 'z']

#sets distance of 'z' according to the measurement plane distances
centroids.loc[centroids['z']==1, 'z'] = 12.5
centroids.loc[centroids['z']==2, 'z'] = 25

#create distance matrics
distances = pdist(centroids.values[:,1:4], metric='euclidean')
dist_matrix = squareform(distances)
dist_matrix = pd.DataFrame(dist_matrix, index=labels, columns=labels)
#dist_matrix.to_csv('Distance_matrix.csv', encoding='ascii')


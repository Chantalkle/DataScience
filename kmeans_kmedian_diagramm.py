import random 
import numpy as np
import pandas as pd
import os
import sklearn
from collections import Counter
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import preprocessing
from collections import defaultdict

from plotly.offline import plot
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
#import seaborn as sns


#from mpl_toolkits.mplot3d import Axes3D

#import streamlit as st   

def numweekdays(day):
    """Only used for Forest Fires DataSet, maps weekdays to numbers"""
    weekdays = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    return weekdays.index(day)+1

def nummonths(month):
    """Only used for Forest Fires DataSet, maps months to numbers"""
    months = ['jan','feb','mar','may','apr','jun','jul','aug','sep','oct','nov','dec']
    return months.index(month)+1

def get_data(data, settype):
    """Reads chosen DataSet (settype) and returns scaled data for k-Means/k-Median"""
    X = pd.read_csv(data, header = 0)

#    print('The data FORMAT is shown as below\n')
#    print(X.head())
#    st.write(X.head())
    X = X.values.tolist()
    if settype == 'Forest Fires':
        for entry in X: 
            entry[2] = nummonths(entry[2])
            entry[3] = numweekdays(entry[3])
    #remove categorial attributes
    if settype == 'Wholesale customers':
        Y = []
        for entry in X:
              Y.append(entry[2:])
            
        X=Y
    if settype == 'Wine':
        Y = [] # Attributes
        
        for entry in X:
            Y.append(entry[1:]) 
        X = Y
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled= min_max_scaler.fit_transform(X)
    X_scaled = X_scaled.tolist()
    return X_scaled

def is_converged(centroids, old_centroids):
    return set([tuple(a) for a in centroids]) == set([tuple(b) for b in old_centroids])

def get_distance(x, c, measuretype):
    """Depending on which distance is given return distance """
 
    if measuretype == 'Manhatten Distance':
        return np.linalg.norm(np.array(x)-np.array(c), ord=1)
    
    elif measuretype == 'Euclidean Distance': 
        return np.linalg.norm(np.array(x)-np.array(c), ord=2)
    
    elif measuretype == 'L5 Distance':
        return np.linalg.norm(np.array(x)-np.array(c), ord=5)
    
    elif measuretype == 'Maximum Distance':
        list_distance = []
        for i in range(len(x)):
            list_distance += [abs(x[i]-c[i])]
        return max(list_distance)

# return: dictionary of lists 
def get_clusters(X, centroids, measuretype):
    clusters = defaultdict(list)
    for x in X:
        # cluster is a num to indicate the # of centroids
        cluster = np.argsort([get_distance(x[:-1], c[:-1], measuretype) for c in centroids])[0]

        clusters[cluster].append(x)
    return clusters

# return: list of np.lists 
def get_centeroids_kmean(old_centroids, clusters):
    new_centroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        if clusters[k]:
            new_centroid = np.mean(clusters[k], axis=0)
            # the class label is determined by majortity vote inside the cluster
            new_centroid[len(clusters[0][0])-1] = Counter([clusters[k][i][-1] for i in range(len(clusters[k]))]).most_common(1)[0][0]
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(old_centroids[k])
    return new_centroids

def get_centeroids_kmed(old_centroids, clusters):
    new_centroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        if clusters[k]:
            new_centroid = np.median(clusters[k], axis=0)
            # the class label is determined by majortity vote inside the cluster
            new_centroid[len(clusters[0][0])-1] = Counter([clusters[k][i][-1] for i in range(len(clusters[k]))]).most_common(1)[0][0]
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(old_centroids[k])
    return new_centroids

# return: tuple (centroids, clusters, iteration)
def find_centers_med(X, K, measuretype):
    old_centroids = random.sample(X, K)
    centroids = random.sample(X, K)
    iteration = 0
    while not is_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = get_clusters(X, centroids, measuretype)
        centroids = get_centeroids_kmed(old_centroids, clusters)
        iteration += 1
    return (centroids, clusters, iteration)

def find_centers_mean(X, K, measuretype):
    old_centroids = random.sample(X, K)
    centroids = random.sample(X, K)
    iteration = 0
    while not is_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = get_clusters(X, centroids, measuretype)
        centroids = get_centeroids_kmean(old_centroids, clusters)
        iteration += 1
    return (centroids, clusters, iteration)

# return purity score 
def get_purity(clusters, centroids, num_instances):
    counts = 0
    for k in clusters.keys():
        labels = np.array(clusters[k])[:, -1]
        counts += Counter(labels).most_common(1)[0][1]
    return float(counts)/num_instances

def get_labels(X, clusters):
    labels = []
    for x in X:
        # find in which cluster x is
        for keys in list(clusters.keys()):
            if x in clusters[keys]:
                labels += [keys]
                break
    return labels

def davis_bouldin(X, labels):
    return metrics.davies_bouldin_score(X, labels)

def cal_hara(X, labels):
    return metrics.calinski_harabasz_score(X, labels)

def sill_co(X,labels):
    return metrics.silhouette_score(X, labels, metric='euclidean')

def rand_sc(Z, labels):
    return adjusted_rand_score(Z, labels)

def kmeans(data, k, distance, output, settype, mean_med):
    X = get_data(data, settype)
    num_instances = len(X)
    if mean_med == 'KMeans':
        centroids, clusters , iteration= find_centers_mean(X, k, distance)
    else:
        centroids, clusters , iteration= find_centers_med(X, k, distance)
    
    #centroids = []
    #for c in best_centroids:
     #   c = c.tolist()
      #  centroids.append(c)
    #best_centroids = centroids
    #print('The best purity score is %f' % best_score)
    #print('It takes %d number of iterations' % best_iteratoin)
    with open(output, 'w') as out:
        for k in clusters.keys():
            out.write('The %d centroid is \n%s\n\n' % (k, centroids[k]))
            out.write('It has following points: \n')
            for pt in clusters[k]:
                out.write('%s\n' % pt)
            out.write('\n\n\n\n')
    

    labels= get_labels(X, clusters)
    davies = davis_bouldin(X, labels)
    silhouette = sill_co(X, labels)
    calinski = cal_hara(X, labels)
    if dataset == 'Wine':
        C = pd.read_csv(data, header = 0)
        C = C.values.tolist()
        Z = [item[0] for item in C]
        rand = rand_sc(Z,labels)
    else:
        rand = 0
    
#    purity= get_purity(clusters, centroids, num_instances)
    return(davies, silhouette, calinski, rand)

    

    
dataset = input("Chose Dataset by number: \n 1. Wholesale customers, 2. Wine, 3. Forest Fires, 4. Heart failure clinical records \n")
m = input('Choose Algorithem by number: \n 1. KMeans or 2. KMedian \n')
if m == '1':
    mean_med = 'KMeans'
elif m == '2':
    mean_med = 'KMedian'

k = input("Choose up to which k (a natural number bigger or equal to 2) you want to evaluate your dataset: \n")
k = int(k)

while k == 1:
    print('Please enter a value k bigger or equal to 2.')
    k = input("Choose up to which k (a natural number bigger or equal to 2) you want to evaluate your dataset: \n")
    k = int(k)

if k not in range(2,11):
    if k > 10:
        print('You choose a relatively big k. The calculations will take some time.\n')
        answer = input('If you want to proceed type Yes if not type anything else:\n')
        if answer != 'Yes':
          k = input("Choose up to which k (a natural number bigger or equal to 2) you want to evaluate your dataset: \n")
          k = int(k) 
    
if dataset == '1':
    dataset = 'Wholesale customers'
elif dataset == '2':
    dataset = 'Wine'
elif dataset == '3':
    dataset = 'Forest Fires'
elif dataset == '4':
    dataset = 'Heart failure clinical records'

davies_list = [[],[],[],[]]
silhouette_list = [[],[],[],[]]
calinski_list = [[],[],[],[]]
rand_list = [[],[],[],[]]
axis = []

for d in range(0,4):
    if d == 0:
        distancemeasure = 'Manhatten Distance'
    elif d == 1:
        distancemeasure = 'Euclidean Distance'
    elif d == 2:
        distancemeasure = 'L5 Distance'
    elif d == 3:
        distancemeasure = 'Maximum Distance'
    

    for x in range(2,k+1):
        if dataset == 'Wholesale customers':
            thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/wholesale.csv'
            (davies, silhouette, calinski, rand)=kmeans(thedata,x,distancemeasure, 'wholesale.out', dataset, mean_med)
        elif dataset == 'Wine':
            thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/wine_data.csv'
            (davies, silhouette, calinski, rand)=kmeans(thedata,x,distancemeasure, 'wine_data.out', dataset, mean_med)
        elif dataset == 'Forest Fires':
            thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/forestfires.csv'
            (davies, silhouette, calinski, rand)=kmeans(thedata,x,distancemeasure, 'forestfire.out', dataset, mean_med)
        elif dataset == 'Heart failure clinical records':
            thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/heart_failure.csv'
            (davies, silhouette, calinski, rand)=kmeans(thedata,x,distancemeasure, 'heart_failure.out', dataset, mean_med)
            
        axis +=[x]
        davies_list[d] += [davies]
        silhouette_list[d] += [silhouette]
        calinski_list[d] += [calinski]
        if dataset == 'Wine':
            rand_list[d] += [rand]
        
    
running = True
while running: 
    des= 'Dataset ' + dataset + ' with '
    des2= ' and with ' + mean_med + '-algorithm'
    desdav = des + 'Davis-Bouldin index' + des2
    dessil = des + 'Silhouette index' + des2
    descal = des + 'Calinski Harabasz index' + des2
    desr = des + 'Rand index' + des2
   
    if dataset == 'Wine':
        ind = input("Choose validatiuon index: DBI or SI or CHI or RI or end  \n")
    else:    
        ind = input("Choose validatiuon index: DBI or SI or CHI or end  \n")
    
    if ind == 'DBI':
        figdav = go.Figure()
        figdav.add_trace(go.Scatter(x=axis, y=davies_list[0],
                            mode='lines+markers',
                            name='Manhatten Distance'))
        figdav.add_trace(go.Scatter(x=axis, y=davies_list[1],
                            mode='lines+markers',
                            name='Euclidean Distance'))
        figdav.add_trace(go.Scatter(x=axis, y=davies_list[2],
                            mode='lines+markers',
                            name='L5 Distance'))
        figdav.add_trace(go.Scatter(x=axis, y=davies_list[3],
                            mode='lines+markers',
                            name='Maximum Distance'))
        figdav.update_layout(title=desdav, xaxis_title='k')
        plot(figdav)
    
    elif ind == 'SI':
        figsil = go.Figure()
        figsil.add_trace(go.Scatter(x=axis, y=silhouette_list[0],
                            mode='lines+markers',
                            name='Manhatten Distance'))
        figsil.add_trace(go.Scatter(x=axis, y=silhouette_list[1],
                            mode='lines+markers',
                            name='Euclidean Distance'))
        figsil.add_trace(go.Scatter(x=axis, y=silhouette_list[2],
                            mode='lines+markers',
                            name='L5 Distance'))
        figsil.add_trace(go.Scatter(x=axis, y=silhouette_list[3],
                            mode='lines+markers',
                            name='Maximum Distance'))
        figsil.update_layout(title=dessil, xaxis_title='k')
        plot(figsil)
        
    elif ind == 'CHI':
        figcal = go.Figure()
        figcal.add_trace(go.Scatter(x=axis, y=calinski_list[0],
                            mode='lines+markers',
                            name='Manhatten Distance'))
        figcal.add_trace(go.Scatter(x=axis, y=calinski_list[1],
                            mode='lines+markers',
                            name='Euclidean Distance'))
        figcal.add_trace(go.Scatter(x=axis, y=calinski_list[2],
                            mode='lines+markers',
                            name='L5 Distance'))
        figcal.add_trace(go.Scatter(x=axis, y=calinski_list[3],
                            mode='lines+markers',
                            name='Maximum Distance'))
        figcal.update_layout(title=descal, xaxis_title='k')
        plot(figcal)
    
    elif (ind == 'RI' and dataset == 'Wine'):
        figr = go.Figure()
        figr.add_trace(go.Scatter(x=axis, y=rand_list[0],
                            mode='lines+markers',
                            name='Manhatten Distance'))
        figr.add_trace(go.Scatter(x=axis, y=rand_list[1],
                            mode='lines+markers',
                            name='Euclidean Distance'))
        figr.add_trace(go.Scatter(x=axis, y=rand_list[2],
                            mode='lines+markers',
                            name='L5 Distance'))
        figr.add_trace(go.Scatter(x=axis, y=rand_list[3],
                            mode='lines+markers',
                            name='Maximum Distance'))
        figr.update_layout(title=desr, xaxis_title='k')
        plot(figr)
    
    elif ind == 'end': 
        running = False
    
    else:
        print("Error!")

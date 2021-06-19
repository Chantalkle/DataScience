import random 
import numpy as np
import pandas as pd
import os
import sklearn
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from collections import defaultdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import re
import json

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import streamlit as st

 
def numweekdays(day):
    weekdays = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    return weekdays.index(day)+1

def nummonths(month):
    months = ['jan','feb','mar','may','apr','jun','jul','aug','sep','oct','nov','dec']
    return months.index(month)+1


def get_data(data, settype):
    
    X = pd.read_csv(data, header = 0)

    print('The data FORMAT is shown as below\n')
    print(X.head())
    st.write(X.head())
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

    return X

def is_converged(centroids, old_centroids):
    return set([tuple(a) for a in centroids]) == set([tuple(b) for b in old_centroids])

# return int: euclidean distance
def get_distance(x, c, measuretype):
    """Depending on which distance is given return distance """
 
    if measuretype == 'Manhatten Distance':
        return np.linalg.norm(np.array(x)-np.array(c), ord=1)
    
    elif measuretype == 'Euclidean Distance': 
        return np.linalg.norm(np.array(x)-np.array(c), ord=2)
    
    elif measuretype == 'L5 Distance':
        return np.linalg.norm(np.array(x)-np.array(c), ord=5)
    
    elif measuretype == 'Maximum Norm':
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


def kmeans(data, k, distance, output, settype, mean_med):
   
    X = get_data(data, settype)
    num_instances = len(X)
    if mean_med == 'KMeans':
        centroids, clusters , iteration= find_centers_mean(X, k, distance)
    else:
        centroids, clusters , iteration= find_centers_med(X, k, distance)
    # store the best records in 5 iterations 
    best_score = 0
    best_centroids = []
    best_clusters =[]
    best_iteratoin = 0
    if mean_med == 0:
        for i in range(5):
            centroids, clusters , iteration= find_centers_mean(X, k, distance)
            purity = get_purity(clusters, centroids, num_instances)
            if purity > best_score:
                best_centroids = centroids
                best_clusters = clusters
                best_score = purity
                best_iteratoin = iteration
    else:
        for i in range(5):
            centroids, clusters , iteration= find_centers_med(X, k, distance)
            purity = get_purity(clusters, centroids, num_instances)
            if purity > best_score:
                best_centroids = centroids
                best_clusters = clusters
                best_score = purity
                best_iteratoin = iteration
         
    centroids = []
    for c in best_centroids:
        c = c.tolist()
        centroids.append(c)
    best_centroids = centroids
    print('The best purity score is %f' % best_score)
    print('It takes %d number of iterations' % best_iteratoin)
    with open(output, 'w') as out:
        for k in best_clusters.keys():
            out.write('The %d centroid is \n%s\n\n' % (k, best_centroids[k]))
            out.write('It has following points: \n')
            for pt in clusters[k]:
                out.write('%s\n' % pt)
            out.write('\n\n\n\n')
    
    #evaluation = input("Do you want to evalate? 0: No 1: Yes \n")
    
    if evaluation == 'Yes':
        labels = get_labels(X, clusters)
        #print('DBI: ', davis_bouldin(X, labels))
        #print('CHI: ', cal_hara(X, labels))
        #print('SC: ', sill_co(X, labels))
        st.write('DBI: ', davis_bouldin(X, labels))
        st.write('CHI: ', cal_hara(X, labels))
        st.write('SC: ', sill_co(X, labels))
        
    else:
        pass
    
    textfile = open('file.out', 'w')
    for k in best_clusters.keys():
        for element in clusters[k]:
            textfile.write(str(element + [k])+ '\n')
    textfile.close()
    get_visu(dataset)
    
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
    return sklearn.metrics.davies_bouldin_score(X, labels)

def cal_hara(X, labels):
    return metrics.calinski_harabasz_score(X, labels)

def sill_co(X,labels):
    return metrics.silhouette_score(X, labels, metric='euclidean')

def get_visu(dataset):
    
    if dataset == 'Wine':
        columns = '14'
    if dataset == 'Wholesale customers':
        columns = '6'
    if dataset == 'Forest Fires':
        columns = '13'
    if dataset == 'Heart failure clinical records':
        columns = '13'

    dataframe1 = pd.read_csv("file.out", header = None)
    dataframe1.to_csv('file.out.csv', index = None)
    thedata = pd.read_csv('file.out.csv', header = 0)
    print(thedata.head())
    thedata[columns] = pd.to_numeric(thedata[columns].astype(str).str[:-1], errors='coerce')
    target = thedata[columns]
    print(target.head())
    attributes = thedata.loc[:, thedata.columns != '0']
    print(attributes.head())
    attributes = attributes.loc[:, attributes.columns != columns]
    print(attributes.head())
        
    tsne = TSNE(n_components = 3, verbose = 1, random_state=123)
    z = tsne.fit_transform(attributes)
        
    df = pd.DataFrame()
    df["y"] = target
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    df["comp-3"] = z[:,2]
        
    # axes instance
    fig2 = plt.figure(figsize=(5,5))
    ax = Axes3D(fig2)
    fig2.add_axes(ax)
        
    # get colormap from seaborn
    #cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
        
    # plot
    sc = ax.scatter(xs = df["comp-1"], ys = df["comp-2"], zs= df["comp-3"], c= df["y"] , marker='o', alpha=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
        
    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
        
    # save
    #plt.savefig("scatter_hue", bbox_inches='tight') 
    st.write(fig2)
        
    tsne1 = TSNE(n_components = 3, verbose = 1, random_state=123)
    z1 = tsne1.fit_transform(attributes)
        
    df1 = pd.DataFrame()
    df1["y"] = target
    df1["comp-1"] = z1[:,0]
    df1["comp-2"] = z1[:,1]
        
    fig1 = plt.figure(figsize=(5,5))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df1.y.tolist(), palette="deep",
                    data=df1).set(title= dataset +" T-SNE projection") 
        
    #plt.show()     
    st.write(fig1) 

  
###################################--Web App--####################################################
st.title('kmeans clustering')

path = os.path.abspath(".")

mean_med = st.selectbox(
    "Choose Algorithem",
    ('KMeans', 'KMedian'))

dataset = st.selectbox(
    "Choose Dataset",
    ('Wholesale customers', 'Heart failure clinical records', 
     'Forest Fires', 'Wine'))

distancemeasure = st.selectbox(
    "Choose distancemeasure",
    ('Manhatten Distance', 'Euclidean Distance', 
     'L5 Distance', 'Maximum Norm'))

k = st.slider('choose k', 1, 100, 5)

evaluation = st.selectbox(
    "Do you want to evalate?",
    ('Yes', 'No'))


if st.button("Start"):
    st.write("Choosen Data Set: " + str(dataset))
    st.write("Choosen Distance: " + str(distancemeasure))
    st.write("Choosen k = " + str(k))
    if dataset == 'Wholesale customers':
         thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/wholesale.csv'
         kmeans(thedata,k,distancemeasure, 'wholesale.out', dataset, mean_med)
    elif dataset == 'Wine':
         thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/wine_data.csv'
         kmeans(thedata,k,distancemeasure, 'wine_data.out', dataset, mean_med)
    elif dataset == 'Forest Fires':
        thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/forestfires.csv'
        kmeans(thedata,k,distancemeasure, 'forestfire.out', dataset, mean_med)
    elif dataset == 'Heart failure clinical records':
        thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/heart_failure.csv'
        kmeans(thedata,k,distancemeasure, 'heart_failure.out', dataset, mean_med)
else: 
    st.write("Ready to calculate!")
    

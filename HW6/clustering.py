import sys
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from pylab import rcParams
from matplotlib import pyplot as plt

N_Animal = 50
N_Feature = 85

m_features = np.zeros((N_Animal, N_Feature))
dict_label = {}
list_label = []

def k_means(X, dict_label, n_clus = 10):
    d_cluster = dict()
    kmeans = KMeans(init='k-means++', n_clusters=n_clus, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    for i in range(N_Animal):
        c = int(labels[i])
        if not c in d_cluster:
            d_cluster[c] = [dict_label[i+1], ]
        else:
            d_cluster[c].append(dict_label[i+1])
    print d_cluster

def dendro(X, arr_label):
    rcParams['figure.figsize'] = 5, 10
    # generate the linkage matrix
    Z = linkage(X, 'ward')
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('sample index')
    plt.xlabel('distance')
    dendrogram(
                Z,
                orientation = 'right',
                labels = arr_label,
              )
    plt.show()

with open('Animals_with_Attributes/classes.txt', 'r') as f:
    for line in f:
        animal_id, animal_name = line.strip().split()
        dict_label[int(animal_id)] = animal_name
        list_label.append(animal_name)

with open('Animals_with_Attributes/predicate-matrix-continuous.txt', 'r') as f:
    count = 0
    for line in f:
        m_features[count] = np.array(map(float, line.strip().split()))
        count += 1

#k_means(m_features, dict_label)
#dendro(m_features, list_label)

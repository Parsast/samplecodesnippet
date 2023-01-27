# kmeans algorithm to cluster NIPS papers based on the words used in them 

import numpy as np
import xlrd
import sys
import csv
from sklearn import preprocessing
from sklearn.cluster import KMeans
import math
import copy
import matplotlib.pyplot as plt

datasetfile = sys.argv[1]
K = int(sys.argv[2])
outputfile = sys.argv[3]
outstream = open(outputfile,'w')

def preprocesss(rawmatrx):
    featmatrix = preprocessing.scale(rawmatrx)
    return featmatrix


def printclusters(clusters,outstream):
    outstream.write("Customer ID")
    outstream.write("\n")
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            outstream.write(str(clusters[i][j]))
            outstream.write("          ")
            outstream.write("\n")
        outstream.write("\n\n\n")
    outstream.write("===============================================================================\n\n")

def kmeans(featmatrix,centroids,clusterdict):
    # sse = 12345
    # diff = 123456
    threshold = 0.001
    counter = 1
    resul = 67
    # sse = 100
    # resul >  threshold
    while( resul >  threshold ):
        # prevsse = sse;
        # print (centroids)
        clusterdict = allocatecentroids(centroids,featmatrix,clusterdict)
        prevcentroids = copy.deepcopy(centroids)
        centroids = updatecentroids(centroids,featmatrix,clusterdict)
        # print (centroids)
        curcentroids = copy.deepcopy(centroids)
        resul = np.zeros((1,len(centroids[0])))
        for index in range(len(prevcentroids)):
            # print(prevcentroids[index] - curcentroids[index])
            resul = ((prevcentroids[index] - curcentroids[index]) ** 2) + resul
        resul = np.sum(resul)
        # print("result is %f ", resul)
        sse = computesse(centroids,featmatrix,clusterdict)
        # diff = prevsse - sse
        counter +=1
    # print(" no of iterations %oi", counter)
    clusters = []
    for i in range(centroids.shape[0]):
        clusters.append([])
    for  sampleidx, clidx in clusterdict.items():
        clusters[clidx].append(sampleidx)
    return clusters,sse




def initializecentroids(numclusts,dimensionality,featmatrix):
        centroidindexak = np.random.random_integers(0, featmatrix.shape[0] -1 ,size= numclusts)
        iak = 0
        centroids = np.empty((numclusts,dimensionality))
        for indi in centroidindexak:
           centroids[iak,:] = featmatrix[indi,:]
           iak +=1
        return centroids

    
def allocatecentroids(centroids,featmatrix,clusterdict):
    for sampleidx in range(featmatrix.shape[0]):
        distance = []
        for centidx in range(centroids.shape[0]):
            dist = (featmatrix[sampleidx,:] - centroids[centidx,:])
            dist = (np.sum(dist ** 2))
            distance.append(float(dist))
        clusterdict[sampleidx] = distance.index(min(distance))
    return clusterdict

def updatecentroids(centroids,featmatrix,clusterdict):
    means = np.zeros(centroids.shape)
    count = [0] * centroids.shape[0]
    for sample,clusidx in clusterdict.items():
        means[clusidx,:] += featmatrix[sample,:]
        count[clusidx] +=1
    for ind in range(len(count)):
        if count[ind] != 0:
            centroids[ind,:] = (means[ind,:] / float(count[ind]))
    return centroids
    
def computesse(centroids,featmatrix,clusterdict):
    summation = 0.0
    dimensionality = featmatrix.shape[1]
    for sampidx,clustidx in clusterdict.items():
        for dimension in range(dimensionality):
            distak = float(float(centroids[clustidx,dimension] - featmatrix[sampidx,dimension]) **2)
            summation +=  distak
    # sse = float(np.sum(summation))
    return summation



def main():
    f = open(datasetfile,'rb')
    reader = csv.reader(f)
    cols = []
    for col in reader:
        cols.append(col)
    labels = cols[0][1:]
    dimensionality = len(cols) - 1 
    nosamples = len(cols[0]) - 1
    featmatrix = np.empty((nosamples,dimensionality))
    for i in range(1,len(cols)):
        featmatrix[:,i-1] = cols[i][1:]
    # print (" hi")
    featmatrix = np.transpose(featmatrix)
    featmatrix = preprocesss(featmatrix)
    dimensionality = featmatrix.shape[1]
    nosamples = featmatrix.shape[0]
    maxak = 20
    klist = list(range(K,maxak))
    sselist = []
    flagak = False
    for numclusts in klist:
        # doing 5 different initializations for each value of k 
        outstream = open(outputfile,'a')
        sseinitialization = []
        for i in range(5):
            centroids = initializecentroids(numclusts,dimensionality,featmatrix)
            initval = [None] * nosamples
            clusterdict = dict(zip(range(nosamples),initval))
            clusters,sse = kmeans(featmatrix,centroids,clusterdict)
            sseinitialization.append(sse)
        sse = float(min(sseinitialization))
        sselist.append(sse)
        print("sse in this iteration wihth k = %i is  %f",numclusts,sse)
        outstream.write("This is the result of clustering for K = " + str(numclusts) + " corresponding to sse " + str(sse) + "\n\n\n")
        printclusters(clusters,outstream)
        outstream.close()
    plt.plot(range(2,maxak), sselist)
    plt.xlabel("K, number of clusters")
    plt.ylabel("SSE , sum of squares error")
    plt.show()
    # print("hi")
main()

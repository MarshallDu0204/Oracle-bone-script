import os, shutil
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def get_file_name(path):

    paths = os.listdir(path)
    pathList = []
    for element in paths:
        tempPath = path+"/"+element
        pathList.append(tempPath)

    return pathList

def knn_detect(file_list, cluster_nums, randomState=None):
    features = []
    files = file_list

    sift = cv2.xfeatures2d.SIFT_create()
    for file in files:
        #print(file)
        img = cv2.imread(file)

        nhist = cv2.calcHist([img], [0,1], None, [96,96], [0.0, 255.0,0.0, 255.0])
        
        nhist = cv2.normalize(nhist, nhist, 0, 255, cv2.NORM_MINMAX).flatten()
        
        features.append(nhist)

    input_x = np.array(features)
    kmeans = KMeans(n_clusters=cluster_nums)
    kmeans.fit(input_x)
    value = sum(np.min(cdist(input_x, kmeans.cluster_centers_, 'euclidean'), axis=1)) / input_x.shape[0]
    
    return kmeans.labels_, kmeans.cluster_centers_


def main():
    path_filenames = get_file_name("/opt/system/single")
    #print(path_filenames)
    
    labels, cluster_centers = knn_detect(path_filenames,30)
    i=0
    while i!=30:
        os.mkdir("/opt/system/data/"+str(i))
        i+=1
    i=0
    for label in labels:
        sourcePath = path_filenames[i]
        destPath = "/opt/system/data/"+str(label)+"/"+str(i)+".jpg"
        shutil.copy(sourcePath,destPath)
        i+=1

main()
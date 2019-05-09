import os,shutil
from PIL import Image
from PCV.clustering import hcluster
from numpy import *

path = "C:/Users/24400/Desktop/test/"
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

features = zeros([len(imlist), 512])
for i, f in enumerate(imlist):
    im = array(Image.open(f))
    
    h, edges = histogramdd(im.reshape(-1, 3), 8, normed=True, range=[(0, 255), (0, 255), (0, 255)])
    features[i] = h.flatten()
tree = hcluster.hcluster(features)

clusters = tree.extract_clusters(0.23 * tree.distance)

i=0
for c in clusters:
    elements = c.get_cluster_elements()
    nbr_elements = len(elements)
    os.mkdir("C:/Users/24400/Desktop/testData1/"+str(i))
    for img in elements:
        originPath = path+str(img)+".jpg"
        destPath = "C:/Users/24400/Desktop/testData1/"+str(i)+"/"+str(img)+".jpg"
        shutil.copy(originPath,destPath)
    i+=1
    print(i)
    
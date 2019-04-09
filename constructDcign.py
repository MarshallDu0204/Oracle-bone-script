import os,shutil,stat
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import Augmentor
import cv2
import random

def augment(img):
	i=1
	while i!=95:
		j=1
		while j!=95:
			if img[i][j]==1:
				img[i-1][j]=1
				img[i+1][j]=1
				img[i][j-1]=1
				img[i][j+1]=1
			j+=1
		i+=1


def compressImg(img):
	sample_image = np.asarray(a=img[:, :, 0], dtype=np.uint8)
	return sample_image


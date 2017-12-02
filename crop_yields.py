import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
import sys
from sys import exit
import sklearn
from sklearn import svm
import time
from sklearn.preprocessing import StandardScaler

wd='/Users/lilllianpetersen/Google Drive/science_fair/'

f=open(wd+'data/crop_production_ethiopia.csv')

crops=np.zeros(shape=(57,100,2))
data=[]
nyears=54
nrows=534
ncrops=151

i=-2
for line in f:
    i+=1
    line=line.replace('"','')
    tmp=line.split(',')
    if i==-1:
	header=tmp
	continue
    #crops[
    data+=[tmp]

crop2014=[]
production2014=np.zeros(shape=(81))
icrop=-1
icol=113 # year 2014
for irow in range(len(data)):
    if data[irow][5]=='Production' and data[irow][icol]!='':
		icrop+=1
		production2014[icrop]=data[irow][icol]
		crop2014.append(data[irow][3])


indexSorted=np.argsort(production2014)
indexSorted=np.flip(indexSorted,0)
for icrop in range(20):
    print crop2014[indexSorted[icrop]],production2014[indexSorted[icrop]]/1.e6

year=np.zeros(shape=(nyears))
col=np.zeros(shape=(nyears),dtype=int)
for iyear in range(nyears):
    year[iyear]=1960+iyear
    col[iyear]=7.+2.*(float(iyear))
	
production=np.zeros(shape=(nyears,ncrops))
crop=[]
cropcode=np.zeros(shape=(ncrops))
nCropsSoFar=-1
for irow in range(len(data)):
	if data[irow][5]=='Production':
		currentcrop=int(data[irow][2])
		foundCrop=False
		for icrop in range(nCropsSoFar):
			if cropcode[icrop]==currentcrop:
				foundCrop=True
				break
		if foundCrop==False:
			nCropsSoFar+=1
			icrop=nCropsSoFar
			crop.append(data[irow][3])
			cropcode[icrop]=int(data[irow][2])
		for iyear in range(nyears):
			if data[irow][col[iyear]]!='':
				production[iyear,icrop]=data[irow][col[iyear]]


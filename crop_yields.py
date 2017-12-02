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

crop=[]
production=np.zeros(shape=(100))
icrop=-1
icol=113 # year 2014
for irow in range(len(data)):
    if data[irow][5]=='Production' and data[irow][icol]!='':
	icrop+=1
        production[icrop]=data[irow][icol]
	crop.append(data[irow][3])


indexSorted=np.argsort(production)
indexSorted=np.flip(indexSorted,0)
ncrops=icrop
for icrop in range(ncrops):
    print crop[indexSorted[icrop]],production[indexSorted[icrop]]/1.e6

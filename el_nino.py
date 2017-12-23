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
from operator import and_

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'

f=open(wddata+'southern_oscillation_index.csv')

nyears=118
elNino=-9999*np.ones(shape=(nyears,12))
plotYear=-9999*np.ones(shape=(nyears,12))

i=-1
for line in f:
    i+=1
    if i==0 or i==1:
        continue
    tmp=line.split(',')
    year=float(tmp[0][0:4])
    y=int(year-1900)
    month=float(tmp[0][4:6])
    m=int(month-1)
    plotYear[y,m]=year+(month+.5)/12
    elNino[y,m]=tmp[1]

Mask=np.ones(shape=(nyears,12))
for y in range(nyears):
    for m in range(12):
        if elNino[y,m]>-100:
            Mask[y,m]=0

elNinoM=np.ma.masked_array(elNino,Mask)

ydata=np.ma.compressed(elNinoM)
x=np.ma.compressed(np.ma.masked_array(plotYear,Mask))

plt.figure(num=None, figsize=(10, 8))
plt.plot(x,ydata,'-*b')
plt.plot(x,[0 for i in range(len(ydata))],'-k')
plt.title('El Nino Index over Time')
plt.xlabel('time')
plt.ylabel('El Nino index')
plt.savefig(wd+'figures/el_nino_over_time',dpi=700)

np.save(wd+'saved_vars/ethiopia/elNino',elNino)
np.save(wd+'saved_vars/ethiopia/elNinoMask',Mask)


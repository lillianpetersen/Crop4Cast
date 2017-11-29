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
#from celery import Celery


wd='/Users/lilllianpetersen/Google Drive/science_fair/'

f=open(wd+'data/san_juan_dengue.csv')

season=np.zeros(shape=(988))
week=np.zeros(shape=(988))
cases=np.zeros(shape=(988))
year=np.zeros(shape=(988))
month=np.zeros(shape=(988))
day=np.zeros(shape=(988))
dayOfYear=np.zeros(shape=(988))
plotYear=np.zeros(shape=(988))
xtime=[]
i=-2
for line in f:
	i+=1
	if i==-1:
		continue

	line=line.replace('"','')
	tmp=line.split(',')
	week[i]=float(tmp[1])
	cases[i]=float(tmp[9])
	
	xtime.append(str(tmp[2]))
        date=xtime[i]
        year[i]=xtime[i][0:4]
        month[i]=xtime[i][5:7]
        day[i]=xtime[i][8:10]
        dayOfYear[i]=(float(month[i])-1)*30+float(day[i])
        plotYear[i]=year[i]+dayOfYear[i]/365.0	

plt.clf()
plt.plot(plotYear,cases,'b')
plt.savefig(wd+'figures/dengue_cases')







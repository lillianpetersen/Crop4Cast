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
i=-2
for line in f:
	i+=1
	if i==-1:
		continue

	line=line.replace('"','')
	tmp=line.split(',')
	year[i]=float(tmp[2][0:4])
	month[i]=float(tmp[2][5:7])
	day[i]=float(tmp[2][8:10])
	week[i]=float(tmp[1])
	cases[i]=float(tmp[9])


plt.plot(cases)
plt.savefig(wd+'figures/dengue_cases')

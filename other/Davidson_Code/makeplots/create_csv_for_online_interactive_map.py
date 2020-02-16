import sys
import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
from sys import exit
import sklearn
import time
from sklearn.preprocessing import StandardScaler
from operator import and_
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'
wdvars='/Users/lilllianpetersen/saved_vars/'
wdfigs='/Users/lilllianpetersen/figures/'

nyears=5
#fcsv = open(wd+'espatial/ndvi_2013-2017.csv','w')
fcsv = open(wd+'espatial/simple_ndvi_2013-2017.csv','w')

for icountry in range(47):

	f=open(wddata+'africa_latlons.csv')
	for line in f:
		tmp=line.split(',')
		if tmp[0]==str(icountry+1):
			country=tmp[1]
			sName=country

	ndviAvg=np.load(wdvars+sName+'/ndviAvg.npy')
	eviAvg=np.load(wdvars+sName+'/eviAvg.npy')
	ndwiAvg=np.load(wdvars+sName+'/ndwiAvg.npy')	

	if ndviAvg.shape[0]==1:
		ndviAvg=ndviAvg[0]

	Mask=np.zeros(shape=(ndviAvg.shape))
	for y in range(nyears):
	    for m in range(12):
	        if math.isnan(ndviAvg[y,m])==True:
	            Mask[y,m]=1	
	
	ndviAvg=np.ma.masked_array(ndviAvg,Mask)

	maxMonth=np.zeros(shape=(nyears),dtype=int)
	for y in range(nyears):
		maxM=np.ma.amax(ndviAvg[y])
		maxMonth[y]=np.where(maxM==ndviAvg[y])[0][0]

	mean=0.
	ndviAnom=np.zeros(shape=(nyears))	
	for y in range(nyears):
		if math.isnan(np.ma.amax(ndviAvg[y]))==False and np.ma.is_masked(np.ma.amax(ndviAvg[y]))==False:
			mean+=np.ma.amax(ndviAvg[y])
	mean=mean/float(nyears)

	for y in range(nyears):
		ndviAnom[y]=np.ma.amax(ndviAvg[y])-mean

	if country=='DR Congo':
		country='Democratic Republic of the Congo'
	if country=='Central Africa Republic':
		country='Central African Republic'
	#if country=='Mauritania':
	#	country='Islamic Republic of Mauritania'
	if country=='Tanzania':
		country='United Republic of Tanzania'
	if country=='Libya':
		country='Libyan Arab Jamahiriya'
	print '\n',country

	for y in range(nyears):
		if np.ma.is_masked(ndviAnom[y])==True or math.isnan(ndviAnom[y])==True:
			ndviAnom[y]=0
	
	#fcsv.write(country+','+str(ndviAnom[0,maxMonth[0]]*100)+','+str(ndviAnom[1,maxMonth[1]]*100)+','+str(ndviAnom[2,maxMonth[2]]*100)+','+str(ndviAnom[3,maxMonth[3]]*100)+','+str(ndviAnom[4,maxMonth[4]]*100)+'\n')
	fcsv.write(country+','+str(ndviAnom[0]*100)+','+str(ndviAnom[1]*100)+','+str(ndviAnom[2]*100)+','+str(ndviAnom[3]*100)+','+str(ndviAnom[4]*100)+'\n')
	print maxMonth

fcsv.close()

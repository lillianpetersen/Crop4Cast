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
wddata='/Users/lilllianpetersen/data/'
wdvars='/Users/lilllianpetersen/saved_vars/'
wdfigs='/Users/lilllianpetersen/figures/'

f=open(wddata+'crop_production_ethiopia.csv')

crops=np.zeros(shape=(57,100,2))
data=[]
nyears=54
nrows=533
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
for icrop in range(10):
    print crop2014[indexSorted[icrop]],production2014[indexSorted[icrop]]/1.e6,indexSorted[icrop]

year=np.zeros(shape=(nyears))
col=np.zeros(shape=(nyears),dtype=int)
for iyear in range(nyears):
    year[iyear]=1960+iyear
    col[iyear]=7.+2.*(float(iyear))
	
production=np.zeros(shape=(nyears,ncrops))
area=np.zeros(shape=(nyears,ncrops))
cropYield=np.zeros(shape=(nyears,ncrops))
crop=[]
cropcode=np.zeros(shape=(ncrops),dtype=int)
nCropsSoFar=-1
for irow in range(len(data)):
	if data[irow][5]=='Area harvested':
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
				area[iyear,icrop]=data[irow][col[iyear]]
			if data[irow+1][col[iyear]]!='':
				cropYield[iyear,icrop]=data[irow+1][col[iyear]]
			if data[irow+2][col[iyear]]!='':
				production[iyear,icrop]=data[irow+2][col[iyear]]

cornYield=np.zeros(shape=(nyears))
b=0
for iyear in range(nyears):
	try:
		cornYield[iyear]=float(data[106][col[iyear]])
		cornYield[iyear]=cornYield[iyear]*(200/125520.) # convert ha/hg to bu/acre
	except:
		b+=1
	try:
		cornYield[iyear]=float(data[366][col[iyear]])
		cornYield[iyear]=cornYield[iyear]*(200/125520.) # convert ha/hg to bu/acre
	except:
		b+=1

year=np.arange(1961,2015)
plt.clf()
plt.figure(figsize=[6,5])
plt.plot(year,cornYield,'-*g')
plt.title('Corn Yield, Ethiopia')
plt.xlabel('Year')
plt.ylabel('Corn Yield, Bu/Acre')
plt.grid(True)
plt.savefig(wdfigs+'Ethiopia/corn_yield',dpi=700)

exit()

indexSorted=np.argsort(production[nyears-1,:])
indexSorted=np.flip(indexSorted,0)

year=np.arange(1961,2015)
legendtext=[]
plt.clf()
plt.figure(figsize=[10,10])
for i in range(2,12):
	plt.plot(year,production[:,indexSorted[i]]/1.e7,'-')
	legendtext.append(crop[indexSorted[i]])
plt.legend(legendtext)
plt.title('Crops Production')
plt.xlabel('year')
plt.ylabel('production (millions tons)')
plt.grid(True)
plt.savefig(wdfigs+'Ethiopia/crop_production')

legendtext=[]
plt.clf()
plt.figure(figsize=[10,10])
plt.plot(year,production[:,indexSorted[0]]/1.e7,'-*')
legendtext.append(crop[indexSorted[0]])
plt.legend(legendtext)
plt.title('Cereal Production')
plt.xlabel('year')
plt.ylabel('production (millions tons)')
plt.grid(True)
plt.savefig(wdfigs+'Ethiopia/cereals_production')

plt.clf()
plt.figure(figsize=[10,10])
plt.plot(year,cropYield[:,indexSorted[0]]/1.e4,'-*')
plt.legend(legendtext)
plt.title('Cereal Yield')
plt.xlabel('year')
plt.ylabel('yield (thousands units)')
plt.grid(True)
plt.savefig(wdfigs+'Ethiopia/cereals_yield')

plt.clf()
plt.figure(figsize=[10,10])
plt.plot(year,area[:,indexSorted[0]]/1.e7,'-*')
plt.legend(legendtext)
plt.title('Cereal Area Harvested')
plt.xlabel('year')
plt.ylabel('area (millions hectares)')
plt.grid(True)
plt.savefig(wdfigs+'Ethiopia/cereals_area')

legendtext=[]
plt.clf()
plt.figure(figsize=[10,10])
for i in range(2,9):
	plt.plot(year,(cropYield[:,indexSorted[i]]/cropYield[0,indexSorted[i]]),'-')
	legendtext.append(crop[indexSorted[i]])
plt.legend(legendtext)
plt.title('Crops Yields')
plt.xlabel('year')
plt.ylabel('normalized yield')
plt.grid(True)
plt.savefig(wdfigs+'Ethiopia/crop_yield')

legendtext=[]
plt.clf()
plt.figure(figsize=[10,10])
for i in range(2,9):
	plt.plot(year,area[:,indexSorted[i]]/1.e7,'-')
	legendtext.append(crop[indexSorted[i]])
plt.legend(legendtext)
plt.title('Cereal Area Harvested')
plt.xlabel('year')
plt.ylabel('area (millions hectares)')
plt.grid(True)
plt.savefig(wdfigs+'Ethiopia/crop_area')

savedCropYield=np.zeros(shape=(116,ncrops))
savedCropYield[61:115,:]=cropYield

np.save(wdvars+'ethiopia/production',production)
np.save(wdvars+'ethiopia/cropYield',savedCropYield)
np.save(wdvars+'ethiopia/areaHarvested',area)

np.save(wdvars+'ethiopia/cropYieldBoxAvg.npy',cropYield[:,indexSorted[0]])

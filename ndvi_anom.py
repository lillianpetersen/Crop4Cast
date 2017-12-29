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

#lat=6.439697635729217
#lon=36.830086263085065
lat=41.32227845829797
lon=-84.61415705767553
nyears=16
pixels=64

#ndviClimo=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviClimo.npy')
#climoCounterAll=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/climoCounter.npy')
#ndviMonthAvg=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviMonthAvg.npy')

ndviClimo=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviClimoUnprocessed.npy')
climoCounterAll=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/climoCounterUnprocessed.npy')
ndviMonthAvg=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviMonthAvgUnprocessed.npy')

eviClimo=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/eviClimoUnprocessed.npy')
eviMonthAvg=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/eviMonthAvgUnprocessed.npy')

#ndwiClimo=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndwiClimoUnprocessed.npy')
#ndwiMonthAvg=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndwiMonthAvgUnprocessed.npy')


#np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/climoCounterAllUnprocessed',climoCounterAll)
#np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviMonthAvgUnprocessed',ndviMonthAvg)
#np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviClimoUnprocessed',ndviMonthAvg)


climoCounter=np.zeros(shape=(12,pixels,pixels)) # number of days in every of each month
ndviAnomAllPix=-9999*np.ones(shape=(nyears,12,pixels,pixels))
eviAnomAllPix=-9999*np.ones(shape=(nyears,12,pixels,pixels))
#ndwiAnomAllPix=-9999*np.ones(shape=(nyears,12,pixels,pixels))

for m in range(12):
	if m!=5 and m!=6 and m!=7 and m!=8:
		continue
	for v in range(pixels):
		for h in range(pixels):
		   climoCounter[m,v,h]=np.sum(climoCounterAll[:,m,v,h])

		   ndviClimo[m,v,h]=ndviClimo[m,v,h]/climoCounter[m,v,h]
		   eviClimo[m,v,h]=eviClimo[m,v,h]/climoCounter[m,v,h]
		   #ndwiClimo[m,v,h]=ndwiClimo[m,v,h]/climoCounter[m,v,h]
		 
for y in range(nyears):
	for m in range(12):
		if m!=5 and m!=6 and m!=7 and m!=8:
			continue
		for v in range(pixels):
			for h in range(pixels):
				ndviMonthAvg[y,m,v,h]=ndviMonthAvg[y,m,v,h]/climoCounterAll[y,m,v,h]
				eviMonthAvg[y,m,v,h]=eviMonthAvg[y,m,v,h]/climoCounterAll[y,m,v,h]
				#ndwiMonthAvg[y,m,v,h]=ndwiMonthAvg[y,m,v,h]/climoCounterAll[y,m,v,h]

				ndviAnomAllPix[y,m,v,h]=ndviMonthAvg[y,m,v,h]-ndviClimo[m,v,h]
				eviAnomAllPix[y,m,v,h]=eviMonthAvg[y,m,v,h]-eviClimo[m,v,h]
				#ndwiAnomAllPix[y,m,v,h]=ndwiMonthAvg[y,m,v,h]-ndwiClimo[m,v,h]

plt.clf()
plt.imshow(ndviClimo[7,:,:], vmin=-.6, vmax=.9)
plt.colorbar()
plt.title('ndvi August Climatology Ohio')
plt.savefig(wd+'figures/Ohio/ndviClimo_Aug',dpi=700)

plt.clf()
plt.imshow(eviClimo[7,:,:], vmin=-.6, vmax=.9)
plt.colorbar()
plt.title('evi August Climatology Ohio')
plt.savefig(wd+'figures/Ohio/eviClimo_Aug',dpi=700)

#plt.clf()
#plt.imshow(ndwiClimo[7,:,:], vmin=-.6, vmax=.9)
#plt.colorbar()
#plt.title('ndwi August Climatology Ohio')
#plt.savefig(wd+'figures/Ohio/ndwiClimo_Aug',dpi=700)

ndviAnomAllPixPlot=ndviAnomAllPix
eviAnomAllPixPlot=eviAnomAllPix
#ndwiAnomAllPixPlot=ndwiAnomAllPix
for m in range(12):
	if m<5 or m>8:
		ndviAnomAllPixPlot[:,m,:,:]=0
		eviAnomAllPixPlot[:,m,:,:]=0
		#ndwiAnomAllPixPlot[:,m,:,:]=0

plt.clf()
plt.figure(1,figsize=(3,3))
#plt.plot(np.ma.compressed(ndviAnomAllPix[:,5:8,20,11]),'*-b')
plt.plot(np.ma.compressed(ndviAnomAllPix[:,:,20,11]),'*-b')
plt.ylim(-.25,.25)
plt.title('ndvi Anomaly for pixel 20, 11')
plt.savefig(wd+'figures/Ohio/ndviAnomAllPix_20_11',dpi=700)

plt.clf()
plt.figure(1,figsize=(3,3))
#plt.plot(np.ma.compressed(ndviAnomAllPix[:,5:8,50,30]),'*-b')
plt.plot(np.ma.compressed(ndviAnomAllPix[:,:,50,30]),'*-b')
plt.ylim(-.25,.25)
plt.title('ndvi Anomaly for pixel 50, 30')
plt.savefig(wd+'figures/Ohio/ndviAnomAllPix_50_30',dpi=700)


ndviAnom=np.zeros(shape=(nyears,12))
eviAnom=np.zeros(shape=(nyears,12))
#ndwiAnom=np.zeros(shape=(nyears,12))

for y in range(nyears):
	print y
	for m in range(12):
		if m!=6 and m!=7 and m!=8 and m!=9:
			continue
		c=0.
		for v in range(pixels):
			for h in range(pixels):
				if math.isnan(ndviAnomAllPix[y,m,v,h])==False:
					c+=1
					ndviAnom[y,m]+=np.mean(ndviAnomAllPix[y,m,v,h])
					eviAnom[y,m]+=np.mean(eviAnomAllPix[y,m,v,h])
					#ndwiAnom[y,m]+=np.mean(ndwiAnomAllPix[y,m,v,h])
		ndviAnom[y,m]=ndviAnom[y,m]/c
		eviAnom[y,m]=eviAnom[y,m]/c
		#ndwiAnom[y,m]=ndwiAnom[y,m]/c

plt.clf()
plt.figure(1,figsize=(3,3))
plt.plot(np.ma.compressed(ndviAnom[:,5:8]),'*-b')
plt.title('ndvi Anomaly Ohio')
plt.ylim([-.08,.08])
plt.savefig(wd+'figures/Ohio/ndviAnom',dpi=700)

plt.clf()
plt.figure(1,figsize=(3,3))
plt.plot(np.ma.compressed(eviAnom[:,5:8]),'*-b')
plt.title('evi Anomaly Ohio')
plt.ylim([-.08,.08])
plt.savefig(wd+'figures/Ohio/eviAnom',dpi=700)

#plt.clf()
#plt.figure(1,figsize=(3,3))
#plt.plot(np.ma.compressed(ndwiAnom[:,5:8]),'*-b')
#plt.title('ndwi Anomaly Ohio')
#plt.ylim([-.08,.08])
#plt.savefig(wd+'figures/Ohio/ndwiAnom',dpi=700)


np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviClimo',ndviClimo)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/climoCounter',climoCounter)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviMonthAvg',ndviMonthAvg)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviAnom',ndviAnom)

np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/eviClimo',eviClimo)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/eviMonthAvg',eviMonthAvg)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/eviAnom',eviAnom)

#np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndwiClimo',ndwiClimo)
#np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/climoCounter',climoCounter)
#np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndwiMonthAvg',ndwiMonthAvg)
#np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndwiAnom',ndwiAnom)

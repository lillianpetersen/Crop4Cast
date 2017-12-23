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

lat=6.439697635729217
lon=36.830086263085065
nyears=4
pixels=1024

ndviClimo=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviClimo.npy')
climoCounterAll=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/climoCounter.npy')
ndviMonthAvg=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviMonthAvg.npy')

ndviClimo=ndviClimo[:12]
climoCounterAll=climoCounterAll[0:4,:12]
ndviMonthAvg=ndviMonthAvg[0:4,:12]

np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/climoCounterAllUnprocessed',climoCounterAll)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviMonthAvgUnprocessed',ndviMonthAvg)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviClimoUnprocessed',ndviMonthAvg)


climoCounter=np.zeros(shape=(12,pixels,pixels))
ndviAnomAllPix=-9999*np.ones(shape=(nyears,12,pixels,pixels))

for m in range(12):
    for v in range(pixels):
        for h in range(pixels):
           climoCounter[m,v,h]=np.sum(climoCounterAll[:,m,v,h])
           ndviClimo[m,v,h]=ndviClimo[m,v,h]/climoCounter[m,v,h]
         
for y in range(nyears):
    for m in range(12):
        for v in range(pixels):
            for h in range(pixels):
                ndviMonthAvg[y,m,v,h]=ndviMonthAvg[y,m,v,h]/climoCounterAll[y,m,v,h]
                ndviAnomAllPix[y,m,v,h]=ndviMonthAvg[y,m,v,h]-ndviClimo[m,v,h]

plt.clf()
plt.imshow(ndviClimo[11,:,:])
plt.colorbar()
plt.title('ndvi December Climatology Ethiopia Box')
plt.savefig(wd+'figures/Ethiopia/ndviClimo_Dec',dpi=700)

plt.clf()
plt.figure(1,figsize=(3,3))
plt.plot(np.ma.compressed(ndviAnomAllPix[:,:,200,115]),'*-b')
plt.title('ndvi Anomaly for pixel 200, 115')
plt.ylim([-.08,.08])
plt.savefig(wd+'figures/Ethiopia/ndviAnomAllPix_200_115',dpi=700)

plt.clf()
plt.figure(1,figsize=(3,3))
plt.plot(np.ma.compressed(ndviAnomAllPix[:,:,50,600]),'*-b')
plt.title('ndvi Anomaly for pixel 50, 600')
plt.ylim([-.08,.08])
plt.savefig(wd+'figures/Ethiopia/ndviAnomAllPix_50_600',dpi=700)


ndviAnom=np.zeros(shape=(nyears,12))

for y in range(nyears):
    print y
    for m in range(12):
        c=0.
        for v in range(pixels):
            for h in range(pixels):
                if math.isnan(ndviAnomAllPix[y,m,v,h])==False:
                    c+=1
                    ndviAnom[y,m]+=np.mean(ndviAnomAllPix[y,m,v,h])
        ndviAnom[y,m]=ndviAnom[y,m]/c

np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviClimo',ndviClimo)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/climoCounter',climoCounter)
np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviMonthAvg',ndviMonthAvg)

from netCDF4 import Dataset
from numpy import arange # array module from http://numpy.scipy.org
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylab import *
import csv
from math import sqrt
from sys import exit
import numpy as np
import pickle
import os
from scipy import stats
import matplotlib.lines as lines

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'

folder=wddata+'tamsat_monthly_rainfall/'

nlat=1974
nlon=1894
iBeg=83
iEnd=117
startyear=1983
nyears=iEnd-iBeg
rfe=-9999*np.ones(shape=(nyears,12,nlat,nlon))
ethprecip=-9999*np.ones(shape=(nyears,12,132,132))
ethprecipBoxAvg=-9999*np.ones(shape=(nyears,12))
ethClimo=np.zeros(shape=(12,132,132))
ethClimoBoxAvg=np.zeros(shape=(12))
badMonths=np.ones(shape=(nyears,12))
ethAnom=-9999*np.ones(shape=(nyears,12,132,132))
ethAnomBoxAvg=-9999*np.ones(shape=(nyears,12))

climoCounter=np.zeros(shape=(12))
for y in range(nyears):
    year=str(y+startyear)
    print year
    for m in range(12):
        month=str(m+1)
        try:
            month[1]
        except:
            month='0'+month
        ncfile=folder+year+'/'+month+'/rfe'+year+'_'+month+'.nc'
        try:
            fnc=Dataset(ncfile,'r')
        except:
            print 'file '+ncfile+' does not exist'
            continue
        climoCounter[m]+=1
        badMonths[y,m]=0 # Good
        time=fnc.variables['time']
        lat=fnc.variables['lat']
        lon=fnc.variables['lon']
        rfe[y,m,:,:]=fnc.variables['rfe'][0] # rainfall estimates
        #np.where((lon[:]>=35) & (lon[:]<=40))
        #np.where((lon[:]>=5) & (lon[:]<=10))
        ethprecip[y,m,:,:]=rfe[y,m,641:773,1441:1573] 
        ethprecipBoxAvg[y,m]=np.mean(ethprecip[y,m,:,:])
        ethClimo[m,:,:]+=ethprecip[y,m,:,:]

        #plt.clf()
        #plt.imshow(rfe[0], extent=[np.amin(lon),np.amax(lon),np.amin(lat),nethprecip[y,m,:,:]-ethClimo[m,:,:]p.amax(lat)])
        #plt.colorbar()
        #plt.savefig(wd+'figures/Ethiopia/'+year+'_'+month+'_precip',dpi=700)

for m in range(12):
    ethClimo[m,:,:]=ethClimo[m,:,:]/climoCounter[m]
    ethClimoBoxAvg[m]=np.mean(ethClimo[m,:,:])

for y in range(nyears):
    for m in range(12):
        if np.mean(ethprecip[y,m,:,:])>-500:
            ethAnom[y,m,:,:]=ethprecip[y,m,:,:]-ethClimo[m,:,:]
            ethAnomBoxAvg[y,m]=np.mean(ethAnom[y,m,:,:])

plotYear=np.zeros(shape=(nyears,12))
for y in range(nyears):
    for m in range(12):
        plotYear[y,m]=y+startyear+(m+.5)/12.

plt.clf()
plt.plot(np.ma.compressed(np.ma.masked_array(plotYear,badMonths)),np.ma.compressed(np.ma.masked_array(ethprecipBoxAvg,badMonths)))
plt.title('Ethiopia Precipitation over time')
plt.grid(True)
plt.savefig(wd+'figures/Ethiopia/precip_over_time_tamsat',dpi=700)

plt.clf()
plt.plot(np.arange(1,13),ethClimoBoxAvg)
plt.title('Ethiopia Climatology')
plt.grid(True)
plt.savefig(wd+'figures/Ethiopia/climatology_tamsat',dpi=700)

plt.clf()
plt.plot(np.ma.compressed(np.ma.masked_array(plotYear,badMonths)),np.ma.compressed(np.ma.masked_array(ethAnomBoxAvg,badMonths)))
plt.title('Ethiopia Precipitation Anomaly over time')
plt.grid(True)
plt.savefig(wd+'figures/Ethiopia/precip_anom_over_time_tamsat',dpi=700)


np.save(wd+'saved_vars/ethiopia/PrecipAnomBoxAvg.npy',ethAnomBoxAvg)
exit()

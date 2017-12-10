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

fgridded=open(wddata+'noaa_daily_data/monthly/gridded/gridded_temp.dat')
fonemonth=open(wddata+'noaa_daily_data/monthly/gridded/one_month.dat')


nyears=2018-1880
data=-9999*np.ones(shape=(nyears,12,36,72))
lat=np.zeros(shape=(36))
lon=np.zeros(shape=(72))

#tmp=fonemonth.readline()
#print tmp
#month,year=int(tmp[0:6]),int(tmp[6:12])
#y=year-1880
#m=month-1
#for ilat in range(36):
#    tmp=fonemonth.readline()
#    for ilon in range(72):
#        data[y,m,ilat,ilon]=tmp[6*ilon:6*ilon+6]
#
#plt.clf()
#figure(1,figsize=(15,9))
#plt.imshow(data[y,6],vmin=-150,vmax=250)
#plt.colorbar()
#plt.savefig(wd+'figures/gridded_month')

for line in range(nyears*12):
    tmp=fgridded.readline()
    month,year=int(tmp[0:6]),int(tmp[6:12])
    y=year-1880
    m=month-1
    for ilat in range(36):
        tmp=fgridded.readline()
        for ilon in range(72):
            data[y,m,ilat,ilon]=tmp[6*ilon:6*ilon+6]
            lat[ilat]=ilat*5
            lon[ilon]=ilon*5
lat=-1*(lat-90)
lon=lon-180

ethData=-9999*np.ones(shape=(nyears,12))

for ilat in range(36):
    if lat[ilat]==10:
        for ilon in range(72):
            if lon[ilon]==35:
                ethData[:,:]=data[:,:,ilat,ilon]

Mask=np.ones(shape=(ethData.shape))
for y in range(nyears):
    for m in range(12):
        if ethData[y,m]!=-9999:
            Mask[y,m]=0

numGoodMonths=np.zeros(shape=(nyears))
for y in range(nyears):
    numGoodMonths[y]=np.sum(Mask[y,:])
numGoodMonths[:]=12-numGoodMonths

plotYear=np.zeros(shape=(nyears,12))
for y in range(nyears):
    for m in range(12):
        plotYear[y,m]=y+1880+(m+.5)/12.

plt.clf()
iBeg=1948-1880
iEnd=2017-1880
dataYears=iEnd-iBeg
year=np.arange(iBeg,iEnd)
year=year+1880
plt.plot(np.ma.compressed(plotYear[ethData!=-9999]),np.ma.compressed(ethData[ethData!=-9999]))
badDataPercent=np.sum(Mask[iBeg:iEnd])/float(dataYears*12)*100
plt.title('Ethiopia Gridded Data over Time, Bad Data='+str(round(badDataPercent,2))+'%')
plt.savefig(wd+'figures/Ethiopia/gridded_monthly_temp',dpi=700)

plt.clf()
plt.plot(np.ma.compressed(year),numGoodMonths[iBeg:iEnd])
badDataPercent=np.sum(Mask[iBeg:iEnd])/float(dataYears*12)*100
plt.title('Ethiopia Gridded Data Number of Good Months, Bad Data='+str(round(badDataPercent,2))+'%')
plt.savefig(wd+'figures/Ethiopia/gridded_monthly_good_months',dpi=700)
exit()


#iBeg=1880
#iEnd=2017
#
#
#for y in range(iBeg,iEnd):
#    for m in range(12):
#        format(2i5) month,year
#        for lat = 1 to 36 (85-90N,80-85N,...,80-85S,85-90S)
#           	format(72i6) 180-175W,175-170W,...,170-175E,175-180E

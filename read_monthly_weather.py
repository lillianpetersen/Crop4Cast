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

goodCityRatio=.9
#nstations=35
nstations=2

finv=open(wddata+'noaa_daily_data/monthly/ghcnm_stations/inventory.txt')
fdata=open(wddata+'noaa_daily_data/monthly/ghcnm_stations/data.txt')

lat=np.zeros(shape=(nstations))
lon=np.zeros(shape=(nstations))
city=[]
stationid=[]
iBeg=82
#iBeg=97
iEnd=117
dataYears=iEnd-iBeg
monthlyDataCity=-9999.*ones(shape=(nstations,118,12))
monthlyData=-9999.*ones(shape=(118,12))
k=-1
icity=0
for line in finv:
    tmp=line[0:68]
    countrycode=tmp[0:3]
    if countrycode!='117':
    #if countrycode!='148':
        continue
    k+=1
    stationid.append(tmp[0:11])
    stationidtmp=tmp[0:11]
    if stationid[k]!=stationid[k-1]:
        icity+=1
    lat[icity]=tmp[13:20]
    lon[icity]=tmp[22:30]
    city.append(tmp[38:68])

for line in fdata: #read in the daily data for each closest station
    # initialize tmp variables
    tmp=line[0:115]
    station=tmp[0:11]
    if station==stationid[0] or station==stationid[1]:
        for i in range(2):
            if station==stationid[i]:
                icity=i
        year=int(tmp[11:15])
        y=year-1900
        var=tmp[15:19]
        v=0
        for m in range(12):
            tmp2=tmp[19+8*m:19+8*m+5]
            if tmp2=='   0T' or tmp2=='   0P':
                tmp2='    0'
            if int(tmp2)!=-9999:
                tmp3=(9./5.*float(tmp2)/100.+32.)
            else:
                tmp3=-9999
            monthlyDataCity[icity,y,m]=tmp3

badYear=np.ones(shape=(117))
yAvg=-9999*np.ones(shape=(117))
for y in range(iBeg,iEnd):
    g=0 # number of good months
    j=0    # j is number of good days in month
    b=0    # b is the number of bad days in a month
    for m in range(12):
        if monthlyDataCity[0,y,m]>-100 and monthlyDataCity[1,y,m]>-100:
            j+=1
            monthlyData[y,m]=(monthlyDataCity[0,y,m]+monthlyDataCity[1,y,m])/2

        elif monthlyDataCity[0,y,m]>-100 and monthlyDataCity[1,y,m]<-100:
            j+=1
            monthlyData[y,m]=monthlyDataCity[0,y,m]

        elif monthlyDataCity[0,y,m]<-100 and monthlyDataCity[1,y,m]>-100:
            j+=1
            monthlyData[y,m]=monthlyDataCity[1,y,m]

        else: # dailyData1[v,y,m,d]<-100 and dailyData2[v,y,m,d]<-100 and d!=30:
            b+=1
            monthlyData[y,m]=-9999.
    if j>=10:
        g+=1

    if j>=10:
        badYear[y]=0 # zero means the year is good
        yAvg[y]=np.mean(monthlyData[y,monthlyData[y]!=-9999])
    if j<10:   # make sure the year has atleast 11 good months 
        badYear[y]=1
        yAvg[y]=-9999

if sum(badYear[iBeg:iEnd])/float(dataYears) > 1-goodCityRatio: # if the city has too much bad data
    goodCity=False # True means the city is good

Mask=np.ones(shape=(monthlyData.shape))
for y in range(iBeg,iEnd):
    for m in range(12):
        if monthlyData[y,m]>-100:
            Mask[y,m]=0


plotYear=np.zeros(shape=(117,12))
for y in range(iBeg,iEnd):
    for m in range(12):
        plotYear[y,m]=y+1900+(m+.5)/12.

for icity in range(2):
    for y in range(iBeg,iEnd):
        monthlyDataCity[icity,y,monthlyDataCity[icity,y,:]==-9999]=65

#MaskCity=np.ones(shape=(2,118,112))
#for icity in range(2):
#    for y in range(iBeg,iEnd):
#        for m in range(12):
#            if monthlyDataCity[icity,y,m]>-100:
#                MaskCity[y,m]=0

x=np.ma.compressed(np.ma.masked_array(plotYear[iBeg:iEnd],Mask[iBeg:iEnd]))*10
#x=np.ma.compressed(plotYear[iBeg:iEnd])
ydata=np.ma.compressed(np.ma.masked_array(monthlyData[iBeg:iEnd],Mask[iBeg:iEnd]))
#ydata=np.ma.compressed(monthlyDataCity[1,iBeg:iEnd])
if size(ydata)==0: # don't plot stuff with no data
    exit()

if size(x)==0: # don't plot stuff with no data
    exit()

ydataAvg=np.mean(ydata)
slope,bIntercept=polyfit(x,ydata,1)
yfit=slope*x+bIntercept

plt.clf()
#figure(1,figsize=(9,4))
plt.plot(x,ydata,'--*b',x,yfit,'g')
plt.ylabel('max temp, F')
plt.xlabel('year')
badDataPercent=np.sum(Mask[iBeg:iEnd])/float(dataYears)*10
plt.title('Monthly Ethiopia TMAX, slope='+str(round(slope,2))+' Deg F/Year, bad data='+str(np.round(badDataPercent,2))+'%')
plt.grid(True)
plt.savefig(wd+'figures/Ethiopia/monthly_data',dpi=700)
plt.clf()

plotYear1year=np.arange(1,13)
for y in range(iBeg,iEnd):
    if badYear[y]==True:
        continue
    x=np.ma.compressed(np.ma.masked_array(plotYear1year,Mask[y]))
    #x=np.ma.compressed(plotYear1year)
    ydata=np.ma.compressed(np.ma.masked_array(monthlyData[y],Mask[y]))
    #ydata=np.ma.compressed(monthlyDataCity[1,y])
    if size(ydata)==0: # don't plot stuff with no data
        exit()
    
    if size(x)==0: # don't plot stuff with no data
        exit()

    ydataAvg=np.mean(ydata)
    slope,bIntercept=polyfit(x,ydata,1)
    yfit=slope*x+bIntercept

    plt.plot(x,ydata,'--*b')
    plt.ylabel('max temp, F')
    plt.xlabel('month')
    plt.title('Monthly Ethiopia TMAX, slope='+str(round(slope,2))+' Deg F/Year, bad data='+str(np.round(badDataPercent,2))+'%')
    plt.grid(True)
    plt.savefig(wd+'figures/Ethiopia/monthly_data_year',dpi=700)
plt.clf()




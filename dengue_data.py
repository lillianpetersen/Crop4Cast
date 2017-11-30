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

vlen=2016
hlen=2016
start='1990-01-01'
end='2016-12-31'
nyears=26
country='Puerto_Rico'
makePlots=False
padding = 16
pixels = vlen+2*padding
res = 30

lat=18.0163301232
lon=-66.4310185895
Mask=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/Mask.npy')
ndwiAll=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndwiAll.npy')
plotYearS=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/plotYear.npy')
month=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/month.npy')
year=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/year.npy')
n_good_days=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/n_good_days.npy')+1

ndwiAll=np.ma.masked_array(ndwiAll,Mask)

f=open(wd+'data/san_juan_dengue.csv')
fstation=open(wd+'data/station_data_san_juan.csv')

week=np.zeros(shape=(988))
cases=np.zeros(shape=(988))
year=np.zeros(shape=(988))
month=np.zeros(shape=(988))
day=np.zeros(shape=(988))
dayOfYear=np.zeros(shape=(988))
plotYearD=np.zeros(shape=(988))
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
        plotYearD[i]=year[i]+dayOfYear[i]/365.0	

year=np.zeros(shape=(10178))
month=np.zeros(shape=(10178))
day=np.zeros(shape=(10178))
dayOfYear=np.zeros(shape=(10178))
plotYearW=np.zeros(shape=(10178))
precip=np.zeros(shape=(10178))
xtime=[]
i=-2
for line in fstation:
	i+=1
	if i==-1:
		continue
	tmp=line.split('","')
	xtime.append(str(tmp[2]))
        date=xtime[i]
        year[i]=xtime[i][0:4]
        month[i]=xtime[i][5:7]
        day[i]=xtime[i][8:10]
        dayOfYear[i]=(float(month[i])-1)*30+float(day[i])
        plotYearW[i]=year[i]+dayOfYear[i]/365.0	

	tmp[3]=tmp[3].replace('"','')
	precip[i]=tmp[3]
	
########################
# Average NDWI Monthly #
########################

#ndwiMonths=-9999.*np.ones(shape=(nyears,12,50))
#ndwiMonthsMask=np.zeros(shape=(nyears,12,50),dtype=bool)
ndwiAvg=-9999.*np.ones(shape=(n_good_days))

for k in range(n_good_days):
    ndwiAvg[k]=np.ma.mean(ndwiAll[:,:,k])

precipAvg=np.zeros(shape=((nyears+2)*12-1))
plotYearMonth=np.zeros(shape=((nyears+2)*12-1))
for k in range(10178):
    m=int(month[k])-1
    y=int(year[k]-1990)
    i=12*y+m
    precipAvg[i]+=precip[k]
    plotYearMonth[i]=year[k]+((float(m)+0.5)/12.)


#for v in range(pixels):
#    for h in range(pixels):
#        d=-1*np.ones(shape=(nyears,12),dtype=int)
#        i=-1*np.ones(nyears,dtype=int)
#        for t in range(n_good_days):
#            m=int(month[t])-1
#            y=int(year[t]-int(start[0:4]))
#            d[y,m]+=1
#            i[y]+=1
#            ndwiMonths[y,m,d[y,m]]=ndwiAll[v,h,t]
#            ndwiMonthsMask[y,m,d[y,m]]=Mask[v,h,t]
#
#        for y in range(nyears):
#           for m in range(12):
#               if d[y,m]>0:
#                   if np.ma.is_masked(np.ma.sum(np.ma.masked_array(ndwiMonths[y,m,:d[y,m]+1],
#			ndwiMonthsMask[y,m,:d[y,m]+1]))) == False:
#                      ndwiMedMonths[y,v,h,m]=np.ma.median(np.ma.masked_array(ndwiMonths[y,m,:d[y,m]+1],
#			ndwiMonthsMask[y,m,:d[y,m]+1]))

plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(plotYearMonth,precipAvg,'b-')
ax1.set_xlabel('year')
ax1.set_xlim([1990,2000])
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Monthly Precip (cm)', color='b')
ax1.tick_params('y', colors='b')
ax1.grid(True)

#ax2 = ax1.twinx()
#ax2.plot(plotYearS,ndwiAvg,'r.')
#ax2.set_ylabel('ndwi', color='r')
#ax2.tick_params('y',colors='r')

ax2 = ax1.twinx()
ax2.plot(plotYearD,cases,'r-')
ax2.set_ylabel('cases', color='r')
ax2.tick_params('y',colors='r')
ax2.set_xlim([1990,2000])
ax2.grid(True)
plt.title('San Juan Area: Monthly Precip and Weekly Dengue Data')

fig.tight_layout()
plt.savefig(wd+'figures/dengue_cases_precip')







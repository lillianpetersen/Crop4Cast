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
from operator import and_

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'

###############################################
# Functions
###############################################
def Avg(x):
	'''function to average'''
	xAvg=0.
	for k in range(len(x)):
		xAvg=xAvg+x[k]
	xAvg=xAvg/(k+1)
	return xAvg

def stdDev(x):
	'''function to compute standard deviation'''
	xAvg=np.mean(x)
	xOut=0.
	for k in range(len(x)):
		xOut=xOut+(x[k]-xAvg)**2
	xOut=xOut/(k+1)
	xOut=math.sqrt(xOut)
	return xOut

def Variance(x):
	'''function to compute the variance (std dev squared)'''
	xAvg=np.mean(x)
	xOut=0.
	for k in range(len(x)):
		xOut=xOut+(x[k]-xAvg)**2
	xOut=xOut/(k+1)
	return xOut

def SumOfSquares(x):
	'''function to compute the sum of squares'''
	xOut=0.
	for k in range(len(x)):
		xOut=xOut+x[k]**2
	return xOut

def corr(x,y):
	''' function to find the correlation of two arrays'''
	xAvg=np.mean(x)
	Avgy=np.mean(y)
	rxy=0.
	n=min(len(x),len(y))
	for k in range(n):
		rxy=rxy+(x[k]-xAvg)*(y[k]-Avgy)
	rxy=rxy/(k+1)
	stdDevx=stdDev(x)
	stdDevy=stdDev(y)
	rxy=rxy/(stdDevx*stdDevy)
	return rxy


cropYield=-9999*np.ones(shape=(117))
precipAnom=-9999*np.ones(shape=(117,12))
#ndviAnom=-9999*np.ones(shape=(117,12))

#cropYield[61:115]=np.load(wd+'saved_vars/ethiopia/cropYieldBoxAvg.npy')
#precipAnom[83:117]=np.load(wd+'saved_vars/ethiopia/PrecipAnomBoxAvg.npy')
#tempAnom=np.load(wd+'saved_vars/ethiopia/TempAnomBoxAvg.npy')
#elNino=np.load(wd+'saved_vars/ethiopia/elNino.npy')
#elNinoMask=np.load(wd+'saved_vars/ethiopia/elNinoMask.npy')
#ndviAnom[113:117]=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviAnom.npy')
lat=41.32227845829797
lon=-84.61415705767553
ndviAnom=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviAnom.npy')
eviAnom=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/eviAnom.npy')

#cropYield=np.array([131, 134, 124.1, 155.3, 158.1, 154.2, 145.4, 115.4, 129, 161.3, 140.2, 133, 80.2, 163, 184, 131.6, 163.6])
cropYield=np.array([131, 134, 124.1, 155.3, 158.1, 154.2, 145.4, 115.4, 129, 161.3, 140.2, 133, 80.2, 163, 184, 131.6])


#tempAnom=tempAnom[20:137]
iBeg=0
iEnd=16
nyears=16

varMask=np.ones(shape=(3,nyears,12))
cropMask=np.ones(shape=(nyears))
for y in range(iBeg,iEnd):
	if cropYield[y]>0:
		cropMask[y]=0
	for m in range(12):
		if m>=5 and m<=8:
		#if cropYield[y]>0 and precipAnom[y,m]>-900:
		#	varMask[0,y,m]=0
		#if cropYield[y]>0 and tempAnom[y,m]>-900:
		#	varMask[1,y,m]=0
			if cropYield[y]>0 and ndviAnom[y,m]>-900:
				varMask[0,y,m]=0
			if cropYield[y]>0 and eviAnom[y,m]>-900:
				varMask[1,y,m]=0
			#if cropYield[y]>0 and ndwiAnom[y,m]>-900:
			#	varMask[2,y,m]=0
			


cropYield=np.ma.masked_array(cropYield,cropMask)
#precipAnom=np.ma.masked_array(precipAnom,varMask[0])
#tempAnom=np.ma.masked_array(tempAnom,varMask[1])
#elNino=np.ma.masked_array(elNino,elNinoMask)
ndviAnom=np.ma.masked_array(ndviAnom,varMask[0])
eviAnom=np.ma.masked_array(eviAnom,varMask[1])
#ndwiAnom=np.ma.masked_array(ndwiAnom,varMask[2])

### Plot Yield and NDVI Corr ###
for m in range(6,9):
	cropYield3=np.ma.masked_array(cropYield,varMask[0,:,m])
	Corr=corr(np.ma.compressed(ndviAnom[:,m]),np.ma.compressed(cropYield3))
	
	plt.clf()
	plt.figure(1,figsize=(7,5))
	x=np.ma.compressed(ndviAnom[:,m])
	ydata=np.ma.compressed(cropYield3)
	
	ydataAvg=np.mean(ydata)
	slope,bIntercept=np.polyfit(x,ydata,1)
	yfit=slope*x+bIntercept
	
	plt.plot(x,ydata,'*b',x,yfit,'g-')
	plt.title(str(m)+' ndvi and Crop Yield, Corr='+str(round(Corr,2))+' Slope= '+str(round(slope,2)))
	plt.xlabel('ndvi Anomaly')
	plt.ylabel('crop yield (bu/acre)')
	plt.grid(True)
	plt.savefig(wd+'figures/Ohio/ndvi_yield_corr_'+str(m),dpi=700)

### Plot Yield and EVI Corr ###
for m in range(6,9):
	cropYield3=np.ma.masked_array(cropYield,varMask[0,:,m])
	Corr=corr(np.ma.compressed(eviAnom[:,m]),np.ma.compressed(cropYield3))
	
	plt.clf()
	plt.figure(1,figsize=(7,5))
	x=np.ma.compressed(eviAnom[:,m])
	ydata=np.ma.compressed(cropYield3)
	
	ydataAvg=np.mean(ydata)
	slope,bIntercept=np.polyfit(x,ydata,1)
	yfit=slope*x+bIntercept
	
	plt.plot(x,ydata,'*b',x,yfit,'g-')
	plt.title(str(m)+' evi and Crop Yield, Corr='+str(round(Corr,2))+' Slope= '+str(round(slope,2)))
	plt.xlabel('evi Anomaly')
	plt.ylabel('crop yield (bu/acre)')
	plt.grid(True)
	plt.savefig(wd+'figures/Ohio/evi_yield_corr_'+str(m),dpi=700)
exit()

### Plot Yield and NDWI Corr ###
for m in range(6,9):
	cropYield3=np.ma.masked_array(cropYield,varMask[0,:,m])
	Corr=corr(np.ma.compressed(ndwiAnom[:,m]),np.ma.compressed(cropYield3))
	
	plt.clf()
	plt.figure(1,figsize=(7,5))
	x=np.ma.compressed(ndwiAnom[:,m])
	ydata=np.ma.compressed(cropYield3)
	
	ydataAvg=np.mean(ydata)
	slope,bIntercept=np.polyfit(x,ydata,1)
	yfit=slope*x+bIntercept
	
	plt.plot(x,ydata,'*b',x,yfit,'g-')
	plt.title(str(m)+' ndwi and Crop Yield, Corr='+str(round(Corr,2))+' Slope= '+str(round(slope,2)))
	plt.grid(True)
	plt.savefig(wd+'figures/Ohio/ndwi_yield_corr_'+str(m),dpi=700)

precipSumGS=-9999*np.ones(shape=(nyears))
for y in range(iBeg,iEnd):
	if math.isnan(np.ma.sum(precipAnom[y,0:1]))==False and math.isnan(np.ma.sum(precipAnom[y-1,9:]))==False:
		precipSumGS[y]=np.ma.sum(precipAnom[y,0:1])+np.ma.sum(precipAnom[y-1,9:])
	elif math.isnan(np.ma.sum(precipAnom[y,0:1]))==False:
		precipSumGS[y]=np.ma.sum(precipAnom[y,0:1])
	elif math.isnan(np.ma.sum(precipAnom[y-1,9:]))==False:
		precipSumGS[y]=np.ma.sum(precipAnom[y-1,9:])


tempAvgGS=np.zeros(shape=(nyears))
#ndviAvgGS=np.zeros(shape=(nyears))
elNinoAvgGS=np.zeros(shape=(nyears))
for y in range(iBeg,iEnd):
	goodMonthsGStemp=0
	goodMonthsGSndvi=0
	goodMonthsGSelNino=0
	for m in range(9,12):
		if tempAnom[y-1,m]>-900:
			goodMonthsGStemp+=1
			tempAvgGS[y]+=tempAnom[y-1,m]
		#if ndviAnom[y-1,m]>-900:
		#	goodMonthsGSndvi+=1
		#	ndviAvgGS[y]+=ndviAnom[y-1,m]
	for m in range(5,11):
		if elNino[y-1,m]>-900:
			goodMonthsGSelNino+=1
			elNinoAvgGS[y]+=elNino[y-1,m]
	for m in range(0,1):
		if tempAnom[y,m]>-900:
			goodMonthsGStemp+=1
			tempAvgGS[y]+=tempAnom[y,m]
		#if ndviAnom[y,m]>-900:
		#	goodMonthsGSndvi+=1
		#	ndviAvgGS[y]+=ndviAnom[y,m]
		#if elNino[y,m]>-900:
		#	goodMonthsGSelNino+=1
		#	elNinoAvgGS[y]+=elNino[y,m]
	tempAvgGS[y]=tempAvgGS[y]/goodMonthsGStemp
	#ndviAvgGS[y]=ndviAvgGS[y]/goodMonthsGSndvi
	elNinoAvgGS[y]=elNinoAvgGS[y]/goodMonthsGSelNino

	#precipSsum=np.sum(precipAnom[y,])
	#print 'no data for ',y
	#continue 

Mask1y=np.ones(shape=(5,nyears)) # crop,precip,temp
for y in range(iBeg,iEnd):
	if cropYield[y]>-900:
		Mask1y[0,y]=0
	if precipSumGS[y]>-900 and cropYield[y]>-900:
		Mask1y[1,y]=0
	if tempAvgGS[y]>-900 and cropYield[y]>-900:
		Mask1y[2,y]=0
	#if ndviAvgGS[y]>-900 and cropYield[y]>-900:
	#	Mask1y[3,y]=0
	if elNinoAvgGS[y]>-900 and cropYield[y]>-900:
		Mask1y[4,y]=0

cropYield=np.ma.masked_array(cropYield,Mask1y[0])
precipSumGS=np.ma.masked_array(precipSumGS,Mask1y[1])
tempAvgGS=np.ma.masked_array(tempAvgGS,Mask1y[2])
#ndviAvgGS=np.ma.masked_array(tempAvgGS,Mask1y[3])
elNinoAvgGS=np.ma.masked_array(elNinoAvgGS,Mask1y[4])

Corr=np.zeros(shape=(4))

#### Plot Yield and Precip Corr ###
cropYield1=np.ma.masked_array(cropYield,Mask1y[1])
Corr[0]=corr(np.ma.compressed(precipSumGS),np.ma.compressed(cropYield1))

plt.clf()
plt.figure(1,figsize=(7,5))
x=np.ma.compressed(precipSumGS)
ydata=np.ma.compressed(cropYield1)

ydataAvg=np.mean(ydata)
slope,bIntercept=np.polyfit(x,ydata,1)
yfit=slope*x+bIntercept

plt.plot(x,ydata,'*b',x,yfit,'g-')
plt.title('Season Precip and Crop Yield, corr='+str(round(Corr[0],2))+' Slope= '+str(round(slope,2)))
plt.grid(True)
plt.savefig(wd+'figures/Ethiopia/precip_yield_corr',dpi=700)


#### Plot Yield and Temp Corr ###
cropYield2=np.ma.masked_array(cropYield,Mask1y[2])
Corr[1]=corr(np.ma.compressed(tempAvgGS),np.ma.compressed(cropYield2))

plt.clf()
plt.figure(1,figsize=(7,5))
x=np.ma.compressed(tempAvgGS)
ydata=np.ma.compressed(cropYield2)

ydataAvg=np.mean(ydata)
slope,bIntercept=np.polyfit(x,ydata,1)
yfit=slope*x+bIntercept

plt.plot(x,ydata,'*b',x,yfit,'g-')
plt.title('Season Avg Temp and Crop Yield, Corr='+str(round(Corr[1],2))+' Slope= '+str(round(slope,2)))
plt.grid(True)
plt.savefig(wd+'figures/Ethiopia/temp_yield_corr',dpi=700)


##### Plot Yield and NDVI Corr ###
#cropYield3=np.ma.masked_array(cropYield,Mask1y[3])
#Corr[2]=corr(np.ma.compressed(ndviAvgGS),np.ma.compressed(cropYield3))
#
#plt.clf()
#plt.figure(1,figsize=(7,5))
#x=np.ma.compressed(ndviAvgGS)
#ydata=np.ma.compressed(cropYield3)
#
#ydataAvg=np.mean(ydata)
#slope,bIntercept=np.polyfit(x,ydata,1)
#yfit=slope*x+bIntercept
#
#plt.plot(x,ydata,'*b',x,yfit,'g-')
#plt.title('Season Avg ndvi and Crop Yield, Corr='+str(round(Corr[2],2))+' Slope= '+str(round(slope,2)))
#plt.grid(True)
#plt.savefig(wd+'figures/Ethiopia/ndvi_yield_corr',dpi=700)
#

#### Plot Yield and El Nino Corr ###
cropYield4=np.ma.masked_array(cropYield,Mask1y[4])
Corr[3]=corr(np.ma.compressed(elNinoAvgGS),np.ma.compressed(cropYield4))

plt.clf()
plt.figure(1,figsize=(10,8))
x=np.ma.compressed(elNinoAvgGS)
ydata=np.ma.compressed(cropYield4)

ydataAvg=np.mean(ydata)
slope,bIntercept=np.polyfit(x,ydata,1)
yfit=slope*x+bIntercept

plt.plot(x,ydata,'*b',x,yfit,'g-')
plt.title('Season Avg El Nino and Crop Yield, Corr='+str(round(Corr[3],3))+' Slope= '+str(round(slope,2)))
plt.grid(True)
plt.savefig(wd+'figures/Ethiopia/elNino_yield_corr',dpi=700)


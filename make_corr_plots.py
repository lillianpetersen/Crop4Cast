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
wdvars='/Users/lilllianpetersen/saved_vars/'
wdfigs='/Users/lilllianpetersen/figures/'

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


nyears=16
makePlots=False

precipAnom=-9999*np.ones(shape=(117,12))

cropYieldAll=np.load(wdvars+'cropYield.npy')
cropYieldAll=cropYieldAll[:,100:116,0]

countyName=np.load(wdvars+'countyName.npy')
stateName=np.load(wdvars+'stateName.npy')

ndviAnomAll=np.load(wdvars+'Illinois/ndviAnom.npy')
eviAnomAll=np.load(wdvars+'Illinois/eviAnom.npy')
ndwiAnomAll=np.load(wdvars+'Illinois/eviAnom.npy')

countiesMask=np.zeros(shape=(3143),dtype=bool)
for icounty in range(3143):
	if np.amax(ndviAnomAll[icounty,:])==0:
		countiesMask[icounty]=True

icountyIll=-1
cropYield=np.zeros(shape=(np.sum(1-countiesMask),nyears))
ndviAnom=np.zeros(shape=(np.sum(1-countiesMask),nyears,12))
eviAnom=np.zeros(shape=(np.sum(1-countiesMask),nyears,12))
ndwiAnom=np.zeros(shape=(np.sum(1-countiesMask),nyears,12))
goodCountiesIndex=np.zeros(shape=(np.sum(1-countiesMask)),dtype=int)
for icounty in range(3143):
	if countiesMask[icounty]==False:
		icountyIll+=1
		goodCountiesIndex[icountyIll]=icounty
		cropYield[icountyIll]=cropYieldAll[icounty]
		ndviAnom[icountyIll]=ndviAnomAll[icounty]
		eviAnom[icountyIll]=eviAnomAll[icounty]
		ndwiAnom[icountyIll]=ndwiAnomAll[icounty]

ncounties=icountyIll+1
yieldMask=np.zeros(shape=(ncounties,nyears))
anomMask=np.zeros(shape=(ncounties,nyears,12))
for icounty in range(ncounties):
	for y in range(nyears):
		if cropYield[icounty,y]<1:
			yieldMask[icounty,y]=1
			anomMask[icounty,y,:]=1
		if np.amax(ndviAnom[icounty,y])==0:
			yieldMask[icounty,y]=1
		for m in range(12):
			if m<4 or m>7:
				anomMask[icounty,y,m]=1
			if ndviAnom[icounty,y,m]==0 or math.isnan(ndviAnom[icounty,y,m])==True or ndviAnom[icounty,y,m]<-90:
				anomMask[icounty,y,m]=1


#cropYield=np.ma.masked_array(cropYield,yieldMask)
ndviAnom=np.ma.masked_array(ndviAnom,anomMask)
eviAnom=np.ma.masked_array(eviAnom,anomMask)
ndwiAnom=np.ma.masked_array(ndwiAnom,anomMask)

### Detrend the yield data ###
#cropYieldDet=np.zeros(shape=(cropYield.shape))
#for icounty in range(ncounties):
#	cName=countyName[goodCountiesIndex[icounty]].title()
#	### Plot Normalized Yield ###
#	x=np.ma.masked_array(np.arange(2000,2016),yieldMask[icounty])
#	ydata=np.ma.masked_array(cropYield[icounty],yieldMask[icounty])
#	
#	xPlot=np.ma.compressed(x)
#	ydataPlot=np.ma.compressed(ydata)
#
#	ydataAvg=np.mean(ydataPlot)
#	slope,b=np.polyfit(xPlot,ydataPlot,1)
#	yfit=slope*x+b
#
#	if makePlots:
#		plt.clf()
#		#figure(1,figsize=(9,4))
#		plt.plot(x,ydata,'--*b',x,yfit,'g')
#		plt.ylabel('Yield, Bushels/Acre')
#		plt.xlabel('year')
#		plt.title('Yield: '+cName+', slope='+str(round(slope,2))+' Bu/Acre/Year')
#		plt.grid(True)
#		plt.savefig(wdfigs+'Illinois/'+cName+'_yield_over_time',dpi=700)
#		plt.clf()
#	
#	num=len(yfit)-1
#	
#	cropYieldDet[icounty]=ydata-(slope*x+b)
#	#cropYieldDet[icounty]=cropYieldDet[icounty]+yfit[num]
#	
#	dataAt2015=yfit[num]
#	
#	ydata=np.ma.compressed(np.ma.masked_array(cropYieldDet[icounty],yieldMask[icounty]))
#	x=np.ma.compressed(np.ma.masked_array(x,yieldMask[icounty]))
#	ydataAvg=np.mean(ydata)
#	slope,bIntercept=np.polyfit(x,ydata,1)
#	yfit=slope*x+bIntercept
#	Corr=corr(x,ydata)
#	
#	if makePlots:
#	    plt.clf()
#	    #figure(1,figsize=(9,4))
#	    plt.plot(x,ydata,'*b',x,yfit,'g')
#	    plt.ylabel('Yield')
#	    plt.xlabel('year')
#	    plt.title(cName+' Normalized Corn Yield over time  m='+str(round(slope,3)*100)+' Corr='+str(round(Corr,2)))
#	    plt.grid(True)
#	    plt.savefig(wdfigs+'Illinois/'+cName+'_normalized_yield_over_time',dpi=700)
#	    plt.clf()
	
monthName=['January','Febuary','March','April','May','June','July','August','September','October','November','December']
	
### Plot Yield and NDVI Corr ###
for m in range(4,8):
	cropYield1=np.ma.masked_array(cropYield,anomMask[:,:,m])
	x=np.ma.compressed(cropYield1)
	ydata=np.ma.compressed(ndviAnom[:,:,m])
	Corr=corr(x,ydata)
	
	plt.clf()
	plt.figure(1,figsize=(10,8))
	
	ydataAvg=np.mean(ydata)
	slope,bIntercept=np.polyfit(x,ydata,1)
	yfit=slope*x+bIntercept
	
	plt.plot(x,ydata,'.b',x,yfit,'g-')
	plt.title(monthName[m]+' ndvi and Crop Yield, Corr='+str(round(Corr,2))+' Slope= '+str(round(slope*100,3)))
	plt.ylabel('ndvi Anomaly')
	plt.xlabel('crop yield (bu/acre)')
	plt.ylim([-.01,.01])
	plt.grid(True)
	plt.savefig(wdfigs+'Illinois/ndvi_yield_corr_'+str(m),dpi=700)


### Plot Yield and NDVI Corr ###
for m in range(4,8):
	cropYield1=np.ma.masked_array(cropYield,anomMask[:,:,m])
	x=np.ma.compressed(cropYield1)
	ydata=np.ma.compressed(eviAnom[:,:,m])
	Corr=corr(x,ydata)
	
	plt.clf()
	plt.figure(1,figsize=(10,8))
	
	ydataAvg=np.mean(ydata)
	slope,bIntercept=np.polyfit(x,ydata,1)
	yfit=slope*x+bIntercept
	
	plt.plot(x,ydata,'.b',x,yfit,'g-')
	plt.title(monthName[m]+' evi and Crop Yield, Corr='+str(round(Corr,2))+' Slope= '+str(round(slope*100,3)))
	plt.ylabel('evi Anomaly')
	plt.xlabel('crop yield (bu/acre)')
	plt.ylim([-.01,.01])
	plt.grid(True)
	plt.savefig(wdfigs+'Illinois/edvi_yield_corr_'+str(m),dpi=700)


### Plot Yield and NDVI Corr ###
for m in range(4,8):
	cropYield1=np.ma.masked_array(cropYield,anomMask[:,:,m])
	x=np.ma.compressed(cropYield1)
	ydata=np.ma.compressed(ndwiAnom[:,:,m])
	Corr=corr(x,ydata)
	
	plt.clf()
	plt.figure(1,figsize=(10,8))
	
	ydataAvg=np.mean(ydata)
	slope,bIntercept=np.polyfit(x,ydata,1)
	yfit=slope*x+bIntercept
	
	plt.plot(x,ydata,'.b',x,yfit,'g-')
	plt.title(monthName[m]+' ndwi and Crop Yield, Corr='+str(round(Corr,2))+' Slope= '+str(round(slope*100,3)))
	plt.xlabel('ndwi Anomaly')
	plt.ylabel('crop yield (bu/acre)')
	plt.ylim([-.01,.01])
	plt.grid(True)
	plt.savefig(wdfigs+'Illinois/ndwi_yield_corr_'+str(m),dpi=700)
exit()


#varMask=np.ones(shape=(3,nyears,12))
#cropMask=np.ones(shape=(nyears))
#for y in range(iBeg,iEnd):
#	if cropYield[y]>0:
#		cropMask[y]=0
#	for m in range(12):
#		if m>=5 and m<=8:
#		#if cropYield[y]>0 and precipAnom[y,m]>-900:
#		#	varMask[0,y,m]=0
#		#if cropYield[y]>0 and tempAnom[y,m]>-900:
#		#	varMask[1,y,m]=0
#			if cropYield[y]>0 and ndviAnom[y,m]>-900:
#				varMask[0,y,m]=0
#			if cropYield[y]>0 and eviAnom[y,m]>-900:
#				varMask[1,y,m]=0
#			#if cropYield[y]>0 and ndwiAnom[y,m]>-900:
#			#	varMask[2,y,m]=0
#			
#
#
#cropYield=np.ma.masked_array(cropYield,cropMask)
##precipAnom=np.ma.masked_array(precipAnom,varMask[0])
##tempAnom=np.ma.masked_array(tempAnom,varMask[1])
##elNino=np.ma.masked_array(elNino,elNinoMask)
#ndviAnom=np.ma.masked_array(ndviAnom,varMask[0])
#eviAnom=np.ma.masked_array(eviAnom,varMask[1])
##ndwiAnom=np.ma.masked_array(ndwiAnom,varMask[2])

### Plot Yield and NDVI Corr ###
for m in range(6,9):
	#cropYield3=np.ma.masked_array(cropYield,varMask[0,:,m])
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


import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
import sys
from sys import exit
import sklearn
import time
from sklearn.preprocessing import StandardScaler
from operator import and_
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

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

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

colors = [(.4,0,.6), (0,0,.7), (0,.6,1), (.9,.9,1), (1,.8,.8), (1,1,0), (.8,1,.5), (.1,.7,.1), (.1,.3,.1)]
my_cmap = make_cmap(colors)
my_cmap_r=make_cmap(colors[::-1])

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'
wdvars='/Users/lilllianpetersen/saved_vars/'
wdfigs='/Users/lilllianpetersen/figures/'

makePlots=True
nyears=5


#country='Tunisia'
#cropYield=np.array([[975,1513,912,1050,1200],[70,340,140,100,240]]) 
#crop=['Wheat','Olive Oil']

country='Morocco'
cropYield=np.array([[6934,5116,8064,2731,6250],[2723,1638,3400,620,2000]]) #Wheat
crop=['Wheat','Barley']

#country='Ethiopia'
#cropYield=np.array([[6492,7235,6800,6350,6500],[3925,4232,3500,3900,4200],[3829,4339,3900,3600,3765]])
#crop=['Corn','Wheat','Sorghum']

ndviAnom=np.load(wdvars+country+'/ndviAnom.npy')
eviAnom=np.load(wdvars+country+'/eviAnom.npy')
ndwiAnom=np.load(wdvars+country+'/ndwiAnom.npy')

ndviAvg=np.load(wdvars+country+'/ndviAvg.npy')
eviAvg=np.load(wdvars+country+'/eviAvg.npy')
ndwiAvg=np.load(wdvars+country+'/ndwiAvg.npy')

monthName=['January','Febuary','March','April','May','June','July','August','September','October','November','December']


ndviAvgPlot=np.ma.compressed(ndviAvg)
eviAvgPlot=np.ma.compressed(eviAvg)
ndwiAvgPlot=np.ma.compressed(ndwiAvg)
xtime=np.zeros(shape=(nyears,12))
for y in range(nyears):
	for m in range(12):
		xtime[y,m]=(y+2013)+(m+1.5)/12
xtime=np.ma.compressed(xtime)

### Plot Corn Yield ###
for cp in range(len(crop)):
	ydata=np.ma.compressed(ndviAvg[0])
	
	#ydataAvg=np.mean(ydata)
	#slope,b=np.polyfit(x,ydata,1)
	#yfit=slope*x+b
	
	if makePlots:
	    plt.clf()
	    plt.plot(xtime,ydata,'-*b')
	    plt.ylabel('NDVI Monthly Average')
	    plt.xlabel('year')
	    plt.title(country+' NDVI Monthly Average')
	    plt.grid(True)
	    plt.savefig(wdfigs+country+'/'+crop[cp]+'ndviAvg_over_time',dpi=700)
	    plt.clf()

########################

###########################################
# Monthly NDVI Avg and Crop Yield
###########################################
for cp in range(len(crop)):
	if country=='Ethiopia':
		ydataNDVI=np.zeros(shape=(nyears))
		for y in range(nyears):
			ydataNDVI[y]=np.amax(ndviAvg[0,y,:])
		#Corr=corr(cropYield[cp,1:],ndviAvg[0,:-1,7])
		Corr=corr(cropYield[cp,1:],ydataNDVI[:-1])
	if country=='Morocco':
		Corr=corr(cropYield[cp,:],ndviAvg[0,:,3])
	if country=='Tunisia':
		Corr=corr(cropYield[cp,:],ndviAvg[0,:,4])
	
	plt.clf()
	fig, ax2 = plt.subplots()
	ax1 = ax2.twinx()
	
	ax2.plot(xtime,ndviAvgPlot,'b*-')
	if country=='Ethiopia':
		ax2.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
	if country=='Morocco':
		ax2.set_yticks([0.25,0.3,0.35,.4,.45,.5,.55,.6,.65])
	ax2.set_ylabel('NDVI Monthly Average',color='b')
	ax2.tick_params(axis='y',colors='b')

	if country=='Ethiopia':
		x=np.arange(2014.2,2018.2)
		ydata=cropYield[cp,1:]
	if country=='Morocco':
		x=np.arange(2013.7,2018.7)
		ydata=cropYield[cp,:]
	if country=='Tunisia':
		x=np.arange(2013.7,2018.7)
		ydata=cropYield[cp,:]

	if cp==0:
		ax1.plot(x,ydata,'-^g')
		ax1.tick_params(axis='y',colors='g')
		ax1.set_ylabel(crop[cp]+' Production, Gigatonnes',color='g')
		if country=='Ethiopia':
			ax1.set_yticks([6000,6200,6400,6600,6800,7000,7200])
		if country=='Morocco':
			ax1.set_yticks([0,1000,2000,3000,4000,5000,6000,7000,8000])
	if cp==1:
		ax1.plot(x,ydata,'-^m')
		ax1.tick_params(axis='y',colors='m')
		ax1.set_ylabel(crop[cp]+' Production, Gigatonnes',color='m')
		if country=='Ethiopia':
			ax1.set_yticks([3100,3300,3500,3700,3900,4100])
		if country=='Morocco':
			ax1.set_yticks([4000,5000,6000,7000,8000])
	if cp==2:
		ax1.plot(x,ydata,'-^y')
		ax1.tick_params(axis='y',colors='y')
		ax1.set_ylabel(crop[cp]+' Production, Gigatonnes',color='y')
		if country=='Ethiopia':
			ax1.set_yticks([3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300])
	plt.title(country+': NDVI Monthly Average and '+crop[cp]+' Production, Corr='+str(round(Corr,2)))
	ax2.grid(True)
	plt.savefig(wdfigs+country+'/ndvi_monthlyavg_with_'+crop[cp],dpi=700)


if country=='Ethiopia':
	cp=1
	Corr=corr(cropYield[cp,:],ndviAvg[0,:,7])

	
	plt.clf()
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	
	ax2.plot(xtime,ndviAvgPlot,'b*-')
	ax2.set_ylabel('NDVI Monthly Average',color='b')
	x=np.arange(2013.9,2018.9)
	ax1.plot(x,cropYield[cp,:],'-^m')
	if cp==1:
		if country=='Ethiopia':
			ax1.set_yticks([3300,3500,3700,3900,4100])
	ax1.set_ylabel(crop[cp]+' Production, Gigatonnes',color='m')
	plt.title(country+' NDVI Monthly Average and '+crop[cp]+' Production, Corr='+str(round(Corr,2)))
	ax1.grid(True)
	if country=='Ethiopia':
		ax2.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
	plt.savefig(wdfigs+country+'/ndvi_monthlyavg_with_'+crop[cp],dpi=700)
	
###########################################
# One Month NDVI Avg and Crop Yield
###########################################
for cp in range(len(crop)):
	if country=='Ethiopia':
		x=np.arange(2013,2017)
		xNDVI=np.arange(2013,2018)
		ydata=cropYield[cp,1:]
		ydataNDVI=np.zeros(shape=(nyears))
		for y in range(nyears):
			ydataNDVI[y]=np.amax(ndviAvg[0,y,:])
		#Corr=corr(cropYield[cp,1:],ndviAvg[0,:-1,7])
		Corr=corr(cropYield[cp,1:],ydataNDVI[:-1])
		#corrMonth=monthName[7]
		corrMonth='Max'
	if country=='Morocco':
		x=np.arange(2013,2018)
		ydata=cropYield[cp,:]
		ydataNDVI=np.zeros(shape=(nyears))
		for y in range(nyears):
			ydataNDVI[y]=np.amax(ndviAvg[0,y,:])
		#Corr=corr(cropYield[cp,:],ndviAvg[0,:,3])
		Corr=corr(cropYield[cp,:],ydataNDVI)
		#corrMonth=monthName[3]
		corrMonth='Max'
	if country=='Tunisia':
		x=np.arange(2013,2018)
		xNDVI=np.arange(2013,2018)
		ydata=cropYield[cp,:]
		Corr=corr(cropYield[cp,:],ndviAvg[0,:,4])
		ydataNDVI=ndviAvg[0,:,4]
		corrMonth=monthName[4]

	plt.clf()
	fig, ax2 = plt.subplots()
	ax1 = ax2.twinx()
	
	if country=='Ethiopia':
		ax2.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,.4])
	if country=='Tunisia':
		if cp==1:
			ax1.set_ylim([0,350])
			ax2.set_ylim([0.1,0.25])
		if cp==0:
			ax1.set_ylim([700,1520])
			ax2.set_ylim([0.11,0.25])
	if country=='Morocco':
		ax1.set_ylim([1800,8300])
		ax2.set_ylim([0.48,0.71])
	ax2.set_ylabel('NDVI Monthly Average',color='b')
	ax1.plot(x,ydata,'-*g')
	ax1.tick_params(axis='y',colors='g')
	ax2.tick_params(axis='y',colors='b')
	ax2.plot(xNDVI,ydataNDVI,'-*b')
	if country=='Ethiopia':
		if cp==0:
			ax1.set_yticks([6000,6200,6400,6600,6800,7000,7200])
		if cp==2:
			ax1.set_ylim([3400,4400])
			ax2.set_ylim([0.06,0.40])
	ax1.set_ylabel('Production, Gigatonnes',color='g')
	plt.title(country+': '+corrMonth+' NDVI and '+crop[cp]+' Production, Corr='+str(round(Corr,2)))
	ax1.grid(True)
	ax1.set_xticks([2013,2014,2015,2016,2017])
	plt.savefig(wdfigs+country+'/'+crop[cp]+'_ndvi_monthlyavg_with_ndviAvgPlot'+country,dpi=700)
exit()

###########################################
# Monthly EVI Avg and Crop Yield
###########################################
for cp in range(len(crop)):
	if country=='Ethiopia':
		Corr=corr(cropYield[cp,1:],eviAvg[0,:-1,7])
	if country=='Morocco':
		x=np.arange(2013.7,2018.7)
		ydata=cropYield[cp,:]
	if country=='Tunisia':
		x=np.arange(2013.7,2018.7)
		ydata=cropYield[cp,:]
	
	plt.clf()
	fig, ax2 = plt.subplots()
	ax1 = ax2.twinx()
	
	ax2.plot(xtime,eviAvgPlot,'b*-')
	ax2.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,.4])
	ax2.set_ylabel('EVI Monthly Average',color='b')
	x=np.arange(2014.2,2018.2)
	ax1.plot(x,ydata,'-*g')
	if country=='Ethiopia':
		ax1.set_yticks([6000,6200,6400,6600,6800,7000,7200])
	ax1.set_ylabel('Production, Gigatonnes',color='g')
	plt.title(country+' EVI Monthly Average and '+crop[cp]+' Production, Corr='+str(round(Corr,2)))
	ax1.grid(True)
	plt.savefig(wdfigs+country+'/'+crop+'_evi_monthlyavg_with_eviAvgPlot'+country,dpi=700)

###########################################
# Monthly NDWI Avg and Crop Yield
###########################################
Corr=corr(cropYield[1:],ndwiAvg[0,:-1,7])

plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax2.plot(xtime,ndwiAvgPlot,'b*-')
ax2.set_yticks([-.15,-.1,-.05,0.0,0.05])
ax2.set_ylabel('NDWI Monthly Average',color='b')
x=np.arange(2014.2,2018.2)
ax1.plot(x,cropYield[1:],'-*g')
ax1.set_yticks([6000,6200,6400,6600,6800,7000,7200])
ax1.set_ylabel('Production, Gigatonnes',color='g')
plt.title(country+' NDWI Monthly Average and '+crop+' Production, Corr='+str(round(Corr,2)))
ax1.grid(True)
plt.savefig(wdfigs+country+'/ndwi_monthlyavg_with_ndwiAvgPlot'+country,dpi=700)
exit()


### Plot Corn Yield ###
x=np.arange(2013,2018)
ydata=cropYield

#ydataAvg=np.mean(ydata)
#slope,b=np.polyfit(x,ydata,1)
#yfit=slope*x+b

if makePlots:
    plt.clf()
    plt.plot(x,ydata,'-*g')
    plt.ylabel('Production, Gigatonnes')
    plt.xlabel('year')
    plt.title(country+' '+crop+' Production')
    plt.grid(True)
    plt.savefig(wdfigs+country+'/yield_over_time',dpi=700)
    plt.clf()
########################

### Plot NDVI Anomaly ###
ndviAnomFeb=np.zeros(shape=(nyears))
for y in range(nyears):
    #ndviAnomFeb[y]=np.ma.mean(ndviAnom[0,y,0:2])
    ndviAnomFeb[y]=np.ma.mean(ndviAnom[0,y,6])
	

x=np.arange(2013,2018)
ydata=ndviAnomFeb*1000

xzero=np.arange(2000,3000)
zero=[0 for i in range(1000)]

if makePlots:
    plt.clf()
    plt.plot(xzero,zero,'k',linewidth=2)
    plt.plot(x,ydata,'--*b')
    plt.xlim([2012,2017])
    plt.ylabel('NDVI Anomaly')
    plt.xlabel('year')
    plt.title(country+' NDVI Anomaly')
    plt.grid(True)
    plt.savefig(wdfigs+country+'/ndvi_anom_over_time',dpi=700)
    plt.clf()
########################

### Plot EVI Anomaly ###
eviAnomFeb=np.zeros(shape=(nyears))
for y in range(nyears):
    #eviAnomFeb[y]=np.ma.mean(eviAnom[0,y,0:2])
    eviAnomFeb[y]=np.ma.mean(eviAnom[0,y,6])

x=np.arange(2013,2018)
ydata=eviAnomFeb*1000

if makePlots:
    plt.clf()
    plt.plot(xzero,zero,'k',linewidth=2)
    plt.plot(x,ydata,'--*b')
    plt.xlim([2012,2017])
    plt.ylabel('EVI Anomaly')
    plt.xlabel('year')
    plt.title(country+' EVI Anomaly')
    plt.grid(True)
    plt.savefig(wdfigs+country+'/evi_anom_over_time',dpi=700)
    plt.clf()
########################

### Plot NDWI Anomaly ###
ndwiAnomFeb=np.zeros(shape=(nyears))
for y in range(nyears):
    #ndwiAnomFeb[y]=np.ma.mean(ndwiAnom[0,y,0:2])
    ndwiAnomFeb[y]=np.ma.mean(ndwiAnom[0,y,6])

ydata=ndwiAnomFeb*1000

if makePlots:
    plt.clf()
    plt.plot(xzero,zero,'k',linewidth=2)
    plt.plot(x,ydata,'--*b')
    plt.xlim([2012,2017])
    plt.ylabel('NDWI Anomaly')
    plt.xlabel('year')
    plt.title(country+' NDWI Anomaly')
    plt.grid(True)
    plt.savefig(wdfigs+country+'/ndwi_anom_over_time',dpi=700)
    plt.clf()
########################

#####################################
# crop Yield and NDVI
#####################################
ydata1=cropYield
ydata2=ndviAnomFeb*1000

Corr=corr(ydata1,ydata2)

if makePlots:
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x,ydata1,'-*g')
    ax2.plot(x,ydata2,'-^r')
    #ax1.set_yticks([60,80,100,120,140,160,180,200])
    #ax1.set_yticks([300,500,700,900,1100,1300,1500])
    #ax2.set_yticks([-3,-2,-1,0,1,2,3,4,5])
    #ax2.set_yticks([-15,-10,-5,0,5,10,15])
    ax1.set_ylabel(crop+' Production, Gigatonnes',color='g')
    ax2.set_ylabel('NDVI Anomaly *1000 (avg over '+country+' and Jan and Feb)',color='r')
    ax1.set_xlabel('year')
    ax1.grid(True)
    plt.title(country+' '+crop+' Production and NDVI Anomaly, Corr= '+str(round(Corr,2)))
    plt.savefig(wdfigs+country+'/corn_yield_and_ndvi_anom',dpi=700)
    plt.clf()


#####################################
# crop Yield and EVI
#####################################
ydata1=cropYield
ydata2=eviAnomFeb*1000

Corr=corr(ydata1,ydata2)

if makePlots:
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x,ydata1,'-*g')
    ax2.plot(x,ydata2,'-^r')
    #ax1.set_yticks([60,80,100,120,140,160,180,200])
    #ax1.set_yticks([300,500,700,900,1100,1300,1500])
    #ax2.set_yticks([-3,-2,-1,0,1,2,3,4,5])
    #ax2.set_yticks([-15,-10,-5,0,5,10,15])
    ax1.set_ylabel(crop+' Production, Gigatonnes',color='g')
    ax2.set_ylabel('EVI Anomaly *1000 (avg over '+country+' and Jan and Feb)',color='r')
    ax1.set_xlabel('year')
    ax1.grid(True)
    plt.title(country+' '+crop+' Production and EVI Anomaly, Corr= '+str(round(Corr,2)))
    plt.savefig(wdfigs+country+'/corn_yield_and_evi_anom',dpi=700)
    plt.clf()


#####################################
# crop Yield and NDWI
#####################################
ydata1=cropYield
ydata2=ndwiAnomFeb*1000

Corr=corr(ydata1,ydata2)

if makePlots:
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x,ydata1,'-*g')
    ax2.plot(x,ydata2,'-^r')
    #ax1.set_yticks([60,80,100,120,140,160,180,200])
    #ax1.set_yticks([300,500,700,900,1100,1300,1500])
    #ax2.set_yticks([-3,-2,-1,0,1,2,3,4,5])
    #ax2.set_yticks([-15,-10,-5,0,5,10,15])
    ax1.set_ylabel(crop+' Production, Gigatonnes',color='g')
    ax2.set_ylabel('NDWI Anomaly *1000 (avg over '+country+' and Jan and Feb)',color='r')
    ax1.set_xlabel('year')
    ax1.grid(True)
    plt.title(country+' '+crop+' Production and NDWI Anomaly, Corr= '+str(round(Corr,2)))
    plt.savefig(wdfigs+country+'/corn_yield_and_ndwi_anom',dpi=700)
    plt.clf()








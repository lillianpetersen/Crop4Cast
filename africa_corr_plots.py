import sys
import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
from sys import exit
import sklearn
import time
from collections import Counter
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
fwrite=open(wddata+'max_ndviMonths.csv','w')

makePlots=False
nyears=5
#nyears=4

CropPrediction=np.zeros(shape=(48))
slopesAll=np.zeros(shape=(48))
bIntAll=np.zeros(shape=(48))
maxMonthAll=np.zeros(shape=(48))

for icountry in range(47):
	countryNum=str(icountry+1)
	countryNum='42'
	if countryNum=='26':
		continue
	
	cropDataDir=wddata+'africa_crop/'
	
	f=open(wddata+'africa_latlons.csv')
	for line in f:
	    tmp=line.split(',')
	    if tmp[0]==countryNum:
			country=tmp[1]
			countryl=country.lower()
			countryl=countryl.replace(' ','_')
			countryl=countryl.replace('-','_')
			corrMonth=tmp[5].title()
			corrMonthName=corrMonth
			corrYear=tmp[6]
			harvestMonth=float(tmp[7])+1
			seasons=tmp[8][:-1].split('/')
			twoSeasons=seasons[0]
			if twoSeasons!='no':
				corrMonth1=int(seasons[1])
				corrMonth2=int(seasons[2])
			break

	try:
		ndviAnom2018=np.load(wdvars+country+'/2018/ndviAnom.npy')
		eviAnom2018=np.load(wdvars+country+'/2018/eviAnom.npy')
		ndwiAnom2018=np.load(wdvars+country+'/2018/ndwiAnom.npy')

		ndviAvg2018=np.load(wdvars+country+'/2018/ndviAvg.npy')
		eviAvg2018=np.load(wdvars+country+'/2018/eviAvg.npy')
		ndwiAvg2018Normal=np.load(wdvars+country+'/2018/ndwiAvg.npy')
	except:
		continue
	
	print '\n',country,countryNum

	files = [filename for filename in os.listdir(cropDataDir) if filename.startswith(countryl+'-')]
	crop=[]
	for n in files:
		tmp=n.split('-')
		croptmp=tmp[1].title()
		crop.append(tmp[1].title())
	
	print crop
	
	cropYield=np.zeros(shape=(len(files),nyears))
	for cp in range(len(files)):
		fCropData=open(wddata+'africa_crop/'+countryl+'-'+crop[cp]+'-production.csv')
		j=-1
		k=-1
		for line in fCropData:
			j+=1
			if j==0:
				continue
			tmp=line.split(',')
			year=int(tmp[0].replace('"',''))
			if year<2013 or year>=2013+nyears:
				continue
			k+=1
			cropYield[cp,k]=float(tmp[1].replace('"',''))

	for cp in range(len(crop)):
		if crop[cp]=='Centrifugal_Sugar':
			crop[cp]='Sugar'
		elif crop[cp]=='Milled_Rice':
			crop[cp]='Rice'
	
	### Find Crop Yield Anomaly ###
	cropYieldAnom=np.zeros(shape=(cropYield.shape))
	for cp in range(len(crop)):
		meanYield=np.mean(cropYield[cp,:])
		for y in range(nyears):
			cropYieldAnom[cp,y]=cropYield[cp,y]-meanYield
	
	if np.amax(cropYieldAnom)==0. and np.amin(cropYieldAnom)==0:
		cropYieldAnom[:]=1
	
	########### load variables ###########
	ndviAnom=np.load(wdvars+country+'/ndviAnom.npy')
	eviAnom=np.load(wdvars+country+'/eviAnom.npy')
	ndwiAnom=np.load(wdvars+country+'/ndwiAnom.npy')
	
	ndviAvg=np.load(wdvars+country+'/ndviAvg.npy')
	eviAvg=np.load(wdvars+country+'/eviAvg.npy')
	ndwiAvg=np.load(wdvars+country+'/ndwiAvg.npy')
	######################################

	if nyears==4:
		ndviAvg=ndviAvg[:-1]
		eviAvg=eviAvg[:-1]
		ndwiAvg=ndwiAvg[:-1]
		ndviAnom=ndviAnom[:-1]
		eviAnom=eviAnom[:-1]
		ndwiAnom=ndwiAnom[:-1]
	
	if len(ndviAvg.shape)==3:
		ndviAvg=ndviAvg[0]	
		eviAvg=eviAvg[0]	
		ndwiAvg=ndwiAvg[0]	
		ndviAnom=ndviAnom[0]	
		eviAnom=eviAnom[0]	
		ndwiAnom=ndwiAnom[0]	
	
	monthName=['January','Febuary','March','April','May','June','July','August','September','October','November','December']
	
	if not os.path.exists(wdfigs+country):
		os.makedirs(wdfigs+country)
	
	xtime=np.zeros(shape=(nyears,12))
	for y in range(nyears):
		for m in range(12):
			xtime[y,m]=(y+2013)+(m+.5)/12
	
	MaskAvg=np.zeros(shape=(ndviAvg.shape))
	MaskAnom=np.zeros(shape=(ndviAnom.shape))
	for y in range(nyears):
		for m in range(12):
			if math.isnan(ndviAvg[y,m])==True:
				MaskAvg[y,m]=1
			if math.isnan(ndviAnom[y,m])==True:
				MaskAnom[y,m]=1
	
	ndviAvg=np.ma.masked_array(ndviAvg,MaskAvg)
	eviAvg=np.ma.masked_array(eviAvg,MaskAvg)
	ndwiAvgNormal=np.ma.masked_array(ndwiAvg,MaskAvg)
	ndviAnom=np.ma.masked_array(ndviAnom,MaskAnom)
	eviAnom=np.ma.masked_array(eviAnom,MaskAnom)
	ndwiAnom=np.ma.masked_array(ndwiAnom,MaskAnom)
	xtime=np.ma.masked_array(xtime,MaskAvg)

	ndwiAnom=abs(ndwiAnom)
	ndwiAnom2018=abs(ndwiAnom2018)

	ndwiAvg=np.zeros(shape=(ndwiAvgNormal.shape))
	mean=np.mean(ndwiAvgNormal)
	for y in range(nyears):
		for m in range(12):
			ndwiAvg[y,m]=(1./(ndwiAvgNormal[y,m]/mean))*mean 
	
	ndwiAvg2018=np.zeros(shape=(ndwiAvg2018Normal.shape))
	for m in range(3):
		ndwiAvg2018[m]=(1./(ndwiAvg2018Normal[m]/mean))*mean 
	
	ndviAvgPlot=np.ma.compressed(ndviAvg)
	eviAvgPlot=np.ma.compressed(eviAvg)
	ndwiAvgPlot=np.ma.compressed(ndwiAvg)
	ndviAnomPlot=np.ma.compressed(ndviAnom)
	eviAnomPlot=np.ma.compressed(eviAnom)
	ndwiAnomPlot=np.ma.compressed(ndwiAnom)
	xtime=np.ma.compressed(xtime)
	
	### Plot Corn Yield ###
	for cp in range(len(crop)):
		ydata=np.ma.compressed(ndviAvg)
		
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
		    plt.savefig(wdfigs+country+'/ndviAvg_over_time',dpi=700)
		    plt.clf()
	
	########################
	Corr=np.zeros(shape=(6,len(crop)))
	slope=np.zeros(shape=(6,len(crop)))
	bInt=np.zeros(shape=(6,len(crop)))
	data2017=np.zeros(shape=(6,len(crop)))
	
	
	###########################################
	# Monthly NDVI Avg and Crop Yield
	###########################################
	bar_width=0.2
	
	## Find Max NDVI for each year ##
	ydataNDVI=np.zeros(shape=(nyears))
	ydataNDVIAnom=np.zeros(shape=(nyears))
	maxMonth=np.zeros(shape=(nyears),dtype=int)
	if twoSeasons!='no':
		for y in range(nyears):
			ydatatmp1=np.ma.amax(ndviAvg[y,corrMonth1-1:corrMonth1+2])
			if country=='Rwanda':
				if y!=0:
					ydatatmp2=np.ma.amax(ndviAvg[y-1,corrMonth2-1:corrMonth2+2])
				else:
					ydataNDVI[y]=ydatatmp1
			else:
				ydatatmp2=np.ma.amax(ndviAvg[y,corrMonth2-1:corrMonth2+2])
			ydataNDVI[y]=np.ma.mean([ydatatmp1,ydatatmp2])
	else:
		for y in range(nyears):
			ydataNDVI[y]=np.ma.amax(ndviAvg[y,:])
			maxMonth[y]=np.ma.where(ndviAvg[y,:]==np.ma.amax(ndviAvg[y,:]))[0][0]
		if np.any(maxMonth==10) or np.any(maxMonth==11) or np.any(maxMonth==0) or np.any(maxMonth==1):
			SeasonOverYear=True
			maxMonthWYears=np.zeros(shape=(nyears,2),dtype=int)
			maxtmp=np.amax(ndviAvg[0,:6])
			maxMonthWYears[0,:]=np.ma.where(ndviAvg[:,:]==maxtmp)[0][0],np.ma.where(ndviAvg[:,:]==maxtmp)[1][0]
			for y in range(1,nyears):
				maxtmp1=np.ma.amax(ndviAvg[y-1,6:])
				maxtmp2=np.ma.amax(ndviAvg[y,:6])
				maxtmp=np.ma.amax([maxtmp1,maxtmp2])
				maxMonthWYears[y,:]=np.ma.where(ndviAvg[:,:]==maxtmp)[0][0],np.ma.where(ndviAvg[:,:]==maxtmp)[1][0]

		if SeasonOverYear:
			for y in range(nyears):
				ydataNDVI[y]=ndviAvg[maxMonthWYears[y,0],maxMonthWYears[y,1]]
	
	## Plot NDVI line with Crop bar ##
	for cp in range(len(crop)):
		if corrMonth!='Max':
			maxMonth[:]=float(corrMonth)
			for y in range(nyears):
				ydataNDVI[y]=ndviAvg[y,int(corrMonth)]
		if corrYear=='after':
			Corr[0,cp]=corr(cropYield[cp,1:],ydataNDVI[:-1])
			#x=np.arange(2013.96,2017.96)
			x=np.arange(2013.+(harvestMonth/12)+1,2013.+(harvestMonth/12)+nyears)
			ydata=cropYield[cp,1:]
			slope[0,cp],bInt[0,cp]=np.polyfit(ydataNDVI[:-1],cropYield[cp,1:],1)
			yfit=slope[0,cp]*ydataNDVI[:-1]+bInt[0,cp]
		elif corrYear=='same':
			Corr[0,cp]=corr(cropYield[cp,:],ydataNDVI[:])
			#x=np.arange(2013.7,2018.7)
			x=np.arange(2013.+(harvestMonth/12),2013.+(harvestMonth/12)+nyears)
			ydata=cropYield[cp,:]
			slope[0,cp],bInt[0,cp]=np.polyfit(ydataNDVI[:],cropYield[cp,:],1)
			yfit=slope[0,cp]*ydataNDVI[:]+bInt[0,cp]
		
		if makePlots:
			plt.clf()
			fig, ax2 = plt.subplots()
			ax1 = ax2.twinx()
			ax2.grid(true)
	
			label=crop[cp]+' production'
			ax2.bar(x,ydata,bar_width,color='g',label=label)
			ax2.legend(loc='upper right')
			ax2.tick_params(axis='y',colors='g')
			ax2.set_ylabel(crop[cp]+' production, gigatonnes',color='g')
	
			#if country=='ethiopia':
			#	ax2.set_ylim([6000,7300])
			#if country=='morocco':
			#	ax2.set_yticks([0,1000,2000,3000,4000,5000,6000,7000,8000])
			ax2.set_ylim([np.ma.amin(ydata)*.06,np.ma.amax(ydata)*1.15])
			ax1.set_ylim([np.ma.amin(ndviavgplot)*.9,np.ma.amax(ydatandvi)*1.1])
	
			#if cp==1:
			#	ax2.plot(x,ydata,'-^m')
			#	ax2.tick_params(axis='y',colors='m')
			#	ax2.set_ylabel(crop[cp]+' production, gigatonnes',color='m')
			#	if country=='ethiopia':
			#		ax2.set_yticks([3100,3300,3500,3700,3900,4100])
			#	if country=='morocco':
			#		ax2.set_yticks([4000,5000,6000,7000,8000])
			#if cp==2:
			#	ax2.plot(x,ydata,'-^y')
			#	ax2.tick_params(axis='y',colors='y')
			#	ax2.set_ylabel(crop[cp]+' production, gigatonnes',color='y')
			#	if country=='ethiopia':
			#		ax2.set_yticks([3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300])
			
			ax1.plot(xtime,ndviavgplot,'b*-',label='monthly ndvi')
			ax1.legend(loc='upper left')
			#if country=='ethiopia':
			#	ax1.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
			#if country=='morocco':
			#	ax1.set_yticks([0.25,0.3,0.35,.4,.45,.5,.55,.6,.65])
			#else:
			#	ax1.set_ylim([0,np.ma.amax(ndviavgplot)*1.2])
			ax1.set_ylabel('ndvi monthly average',color='b')
			ax1.tick_params(axis='y',colors='b')
	
			plt.title(country+': ndvi monthly average and '+crop[cp]+' production, corr='+str(round(corr[0,cp],2)))
			plt.savefig(wdfigs+country+'/monthly_ndvi_avg_with_'+crop[cp]+'.jpg',dpi=700)

	
	
	#if country=='Ethiopia':
	#	cp=1
	#	Corr=corr(cropYield[cp,:],ndviAvg[0,:,7])
	#
	#	
	#	plt.clf()
	#	fig, ax1 = plt.subplots()
	#	ax2 = ax1.twinx()
	#	
	#	ax2.plot(xtime,ndviAvgPlot,'b*-')
	#	ax2.set_ylabel('NDVI Monthly Average',color='b')
	#	x=np.arange(2013.9,2018.9)
	#	ax1.plot(x,cropYield[cp,:],'-^m')
	#	if cp==1:
	#		if country=='Ethiopia':
	#			ax1.set_yticks([3300,3500,3700,3900,4100])
	#	ax1.set_ylabel(crop[cp]+' Production, Gigatonnes',color='m')
	#	plt.title(country+' NDVI Monthly Average and '+crop[cp]+' Production, Corr='+str(round(Corr,2)))
	#	ax1.grid(True)
	#	if country=='Ethiopia':
	#		ax2.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
	#	plt.savefig(wdfigs+country+'/ndvi_monthlyavg_with_'+crop[cp],dpi=700)
		
	###########################################
	# One Month NDVI Avg and Crop Yield
	###########################################
	bar_width = 0.27
	for cp in range(len(crop)):
		ydata=cropYield[cp,:]
		#x=np.arange(2013+.14,2018+.14)
		x=np.arange(2013+.14,2013+nyears+.14)
		#xNDVI=np.arange(2013-.14,2018-.14)
		xNDVI=np.arange(2013-.14,2013+nyears-.14)
		Corr[0,cp]=corr(cropYield[cp,:],ydataNDVI)
	
		if corrYear=='after':
			ydata=cropYield[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[0,cp]=corr(cropYield[cp,1:],ydataNDVI[:-1])
			slope[0,cp],bInt[0,cp]=np.polyfit(ydataNDVI[:-1],cropYield[cp,1:],1)
			yfit=slope[0,cp]*ydataNDVI[:-1]+bInt[0,cp]
		if corrMonth!='Max':
			Corr[0,cp]=corr(cropYield[cp,:],ndviAvg[:,int(corrMonth)])
			ydataNDVI=ndviAvg[:,int(corrMonth)]
			corrMonthName=monthName[int(corrMonth)]
			slope[0,cp],bInt[0,cp]=np.polyfit(ydataNDVI[:],cropYield[cp,:],1)
			yfit=slope[0,cp]*ydataNDVI[:]+bInt[0,cp]

		data2017[0,cp]=ydataNDVI[-1]
	
		if makePlots:
			plt.clf()
			fig, ax2 = plt.subplots()
			ax1 = ax2.twinx()
			
			#if country=='Ethiopia':
			#	ax2.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,.4])
			#	if cp==0:
			#		ax1.set_ylim([6000,7300])
			#		#ax1.set_yticks([6000,6200,6400,6600,6800,7000,7200])
			#	if cp==2:
			#		ax1.set_ylim([3500,4450])
			#		ax2.set_ylim([0.06,0.40])
			#if country=='Tunisia':
			#	if cp==1:
			#		ax1.set_ylim([50,350])
			#		ax2.set_ylim([0.1,0.25])
			#	if cp==0:
			#		ax1.set_ylim([700,1520])
			#		ax2.set_ylim([0.11,0.25])
			#if country=='Morocco':
			#	ax1.set_ylim([2100,8300])
			#	ax2.set_ylim([0.48,0.71])
			#else:
			ax1.set_ylim([np.ma.amin(ydata)*0.9,np.ma.amax(ydata)*1.05])
			ax2.set_ylim([np.ma.amin(ydataNDVI)*0.9,np.ma.amax(ydataNDVI)*1.05])
	
	
			ax2.set_ylabel('NDVI Monthly Average',color='b')
			#ax1.plot(x,ydata,'-*g')
			label=crop[cp]+' Production'
			ax1.bar(x,ydata,bar_width,color='g',label=label)
			ax1.tick_params(axis='y',colors='g')
			ax2.tick_params(axis='y',colors='b')
			#ax2.plot(xNDVI,ydataNDVI,'-*b')
			ax2.bar(xNDVI,ydataNDVI,bar_width,color='b',label='Max NDVI')
	
			ax1.set_ylabel('Production, Gigatonnes',color='g')
			plt.title(country+': '+corrMonthName+' NDVI and '+crop[cp]+' Production, Corr='+str(round(Corr[0,cp],2)))
			ax2.grid(True)
			ax1.set_xticks(range(2013,2013+nyears))
			ax1.legend(loc='upper right')
			ax2.legend(loc='upper left')
			plt.savefig(wdfigs+country+'/ndvi_avg_'+crop[cp]+'_'+country+'.jpg',dpi=700)
	
	
	###########################################
	# Monthly EVI Avg and Crop Yield
	###########################################
	
	ydataEVI=np.zeros(shape=(nyears))
	ydataEVIAnom=np.zeros(shape=(nyears))
	if twoSeasons!='no':
		for y in range(nyears):
			ydatatmp1=np.ma.amax(eviAvg[y,corrMonth1-1:corrMonth1+2])
			if country=='Rwanda':
				if y!=0:
					ydatatmp2=np.ma.amax(eviAvg[y-1,corrMonth2-1:corrMonth2+2])
				else:
					ydataEVI[y]=ydatatmp1
			else:
				ydatatmp2=np.ma.amax(eviAvg[y,corrMonth2-1:corrMonth2+2])
			ydataEVI[y]=np.ma.mean([ydatatmp1,ydatatmp2])
	else:
		for y in range(nyears):
			ydataEVI[y]=np.ma.amax(eviAvg[y,:])
			maxMonth[y]=np.ma.where(eviAvg[y,:]==np.ma.amax(eviAvg[y,:]))[0][0]
		if np.any(maxMonth==10) or np.any(maxMonth==11) or np.any(maxMonth==0) or np.any(maxMonth==1):
			SeasonOverYear=True
			maxMonthWYears=np.zeros(shape=(nyears,2),dtype=int)
			maxtmp=np.amax(eviAvg[0,:6])
			maxMonthWYears[0,:]=np.ma.where(eviAvg[:,:]==maxtmp)[0][0],np.ma.where(eviAvg[:,:]==maxtmp)[1][0]
			for y in range(1,nyears):
				maxtmp1=np.ma.amax(eviAvg[y-1,6:])
				maxtmp2=np.ma.amax(eviAvg[y,:6])
				maxtmp=np.ma.amax([maxtmp1,maxtmp2])
				maxMonthWYears[y,:]=np.ma.where(eviAvg[:,:]==maxtmp)[0][0],np.ma.where(eviAvg[:,:]==maxtmp)[1][0]

	corrMonthName=corrMonth
	if corrMonth!='Max':
		ydataEVI=eviAvg[:,int(corrMonth)]
		corrMonthName=monthName[int(corrMonth)]
	
	for cp in range(len(crop)):
		plt.clf()
		fig, ax2 = plt.subplots()
		ax1 = ax2.twinx()
		ax2.grid(True)
	
		ydata=cropYield[cp,:]
		x=np.arange(2013+.14,2013+.14+nyears)
		xEVI=np.arange(2013-.14,2013-.14+nyears)
		Corr[1,cp]=corr(cropYield[cp,:],ydataEVI)
		slope[1,cp],bInt[1,cp]=np.polyfit(ydataEVI[:],cropYield[cp,:],1)
		yfit=slope[1,cp]*ydataEVI[:]+bInt[1,cp]
	
		if corrYear=='after':
			ydata=cropYield[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[1,cp]=corr(cropYield[cp,1:],ydataEVI[:-1])
			slope[1,cp],bInt[1,cp]=np.polyfit(ydataEVI[:-1],cropYield[cp,1:],1)
			yfit=slope[1,cp]*ydataEVI[:-1]+bInt[1,cp]
	
		data2017[1,cp]=ydataEVI[-1]

		if makePlots:
			ax1.set_ylim([np.ma.amin(ydata)*0.9,np.ma.amax(ydata)*1.05])
			ax2.set_ylim([np.ma.amin(ydataEVI)*0.9,np.ma.amax(ydataEVI)*1.05])
	
			#if country=='Ethiopia':
			#	ax1.set_yticks([6000,6200,6400,6600,6800,7000,7200])
			#if country=='Ethiopia':
			#	Corr=corr(cropYield[cp,1:],eviAvg[0,:-1,7])
			#if country=='Morocco':
			#	x=np.arange(2013.7,2018.7)
			#	ydata=cropYield[cp,:]
			#if country=='Tunisia':
			#	x=np.arange(2013.7,2018.7)
			#	ydata=cropYield[cp,:]
			
			
			ax2.bar(xEVI,ydataEVI,bar_width,color='b',label='Max EVI')
			#ax2.plot(xtime,eviAvgPlot,'b*-')
			label=crop[cp]+' Production'
			ax1.bar(x,ydata,bar_width,color='g',label=label)
			#ax1.plot(x,ydata,'-*g')
	
			ax1.set_xticks(range(2013,2013+nyears))
			ax1.legend(loc='upper right')
			ax2.legend(loc='upper left')
	
			ax2.set_ylabel('EVI Monthly Average',color='b')
			ax1.set_ylabel('Production, Gigatonnes',color='g')
			ax1.tick_params(axis='y',colors='g')
			ax2.tick_params(axis='y',colors='b')
	
			plt.title(country+': '+corrMonthName+' EVI and '+crop[cp]+' Production, Corr='+str(round(Corr[1,cp],2)))
			plt.savefig(wdfigs+country+'/evi_avg_'+crop[cp]+'_'+country+'.jpg',dpi=700)
	
	###########################################
	# Monthly NDWI Avg and Crop Yield
	###########################################
	
	ydataNDWI=np.zeros(shape=(nyears))
	ydataNDWIAnom=np.zeros(shape=(nyears))
	if twoSeasons!='no':
		for y in range(nyears):
			ydatatmp1=np.ma.amax(ndwiAvg[y,corrMonth1-1:corrMonth1+2])
			if country=='Rwanda':
				if y!=0:
					ydatatmp2=np.ma.amax(ndwiAvg[y-1,corrMonth2-1:corrMonth2+2])
				else:
					ydataNDWI[y]=ydatatmp1
			else:
				ydatatmp2=np.ma.amax(ndwiAvg[y,corrMonth2-1:corrMonth2+2])
			ydataNDWI[y]=np.ma.mean([ydatatmp1,ydatatmp2])
	else:
		for y in range(nyears):
			ydataNDWI[y]=np.ma.amax(ndwiAvg[y,:])
			if SeasonOverYear:
				ydataNDWI[y]=ndwiAvg[maxMonthWYears[y,0],maxMonthWYears[y,1]]

	corrMonthName=corrMonth

	if corrMonth!='Max':
		ydataNDWI=np.zeros(shape=(nyears))
		ydataNDWI=np.zeros(shape=(nyears))
		for y in range(nyears):
			ydataNDWI[y]=ndwiAvg[y,int(corrMonth)]
		mean=np.mean(ydataNDWI[y])
		for y in range(nyears):
			ydataNDWI[y]=(1./(ydataNDWI[y]/mean))*mean 
		corrMonthName=monthName[int(corrMonth)]

	
	for cp in range(len(crop)):
		plt.clf()
		fig, ax2 = plt.subplots()
		ax1 = ax2.twinx()
	
		ydata=cropYield[cp,:]
		x=np.arange(2013+.14,2013+.14+nyears)
		xNDWI=np.arange(2013-.14,2013-.14+nyears)
		Corr[2,cp]=corr(cropYield[cp,:],ydataNDWI)
		slope[2,cp],bInt[2,cp]=np.polyfit(ydataNDWI[:],cropYield[cp,:],1)
		yfit=slope[2,cp]*ydataNDWI[:]+bInt[2,cp]
	
		if corrYear=='after':
			ydata=cropYield[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[2,cp]=corr(cropYield[cp,1:],ydataNDWI[:-1])
			slope[2,cp],bInt[2,cp]=np.polyfit(ydataNDWI[:-1],cropYield[cp,1:],1)
			yfit=slope[2,cp]*ydataNDWI[:-1]+bInt[2,cp]

		data2017[2,cp]=ydataNDWI[-1]
	
		if makePlots:
			ax1.set_ylim([np.ma.amin(ydata)*0.9,np.ma.amax(ydata)*1.05])
			ax2.set_ylim([np.ma.amin(ydataNDWI)*0.9,np.ma.amax(ydataNDWI)*1.05])
			
			ax2.bar(xNDWI,ydataNDWI,bar_width,color='b',label='Max NDWI')
			label=crop[cp]+' Production'
			ax1.bar(x,ydata,bar_width,color='g',label=label)
	
			ax1.set_xticks(range(2013,2013+nyears))
			ax1.legend(loc='upper right')
			ax2.legend(loc='upper left')
	
			ax2.set_ylabel('NDWI Monthly Average, flipped',color='b')
			ax1.set_ylabel('Production, Gigatonnes',color='g')
			ax1.tick_params(axis='y',colors='g')
			ax2.tick_params(axis='y',colors='b')
	
			plt.title(country+': '+corrMonthName+' NDWI and '+crop[cp]+' Production, Corr='+str(round(Corr[2,cp],2)))
			ax2.grid(True)
			plt.savefig(wdfigs+country+'/ndwi_avg_'+crop[cp]+'_'+country+'.jpg',dpi=700)
	
	####################################################################################
	####################################################################################
	
	###########################################
	# One Month NDVI Anom and Crop Yield
	###########################################
	bar_width = 0.27
	for cp in range(len(crop)):
		ydata=cropYieldAnom[cp,:]
		x=np.arange(2013+.14,2013+nyears+.14)
		xNDVI=np.arange(2013-.14,2013+nyears-.14)
		if twoSeasons!='no':
			for y in range(nyears):
				ydatatmp1=np.ma.amax(ndviAnom[y,corrMonth1-1:corrMonth1+2])
				if country=='Rwanda':
					if y!=0:
						ydatatmp2=np.ma.amax(ndviAnom[y-1,corrMonth2-1:corrMonth2+2])
					else:
						ydataNDVIAnom[y]=ydatatmp1
				else:
					ydatatmp2=np.ma.amax(ndviAnom[y,corrMonth2-1:corrMonth2+2])
				ydataNDVIAnom[y]=np.ma.mean([ydatatmp1,ydatatmp2])
		else:
			for y in range(nyears):
				ydataNDVIAnom[y]=ndviAnom[y,maxMonth[y]]
				if SeasonOverYear:
					ydataNDVIAnom[y]=ndviAnom[maxMonthWYears[y,0],maxMonthWYears[y,1]]

		if corrMonth!='Max':
			ydataNDVIAnom=ndviAnom[:,int(corrMonth)]

		Corr[3,cp]=corr(cropYieldAnom[cp,:],ydataNDVIAnom)
		slope[3,cp],bInt[3,cp]=np.polyfit(ydataNDVIAnom[:],cropYield[cp,:],1)
		yfit=slope[3,cp]*ydataNDVIAnom[:]+bInt[3,cp]
	
		if corrYear=='after':
			ydata=cropYieldAnom[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[3,cp]=corr(cropYieldAnom[cp,1:],ydataNDVIAnom[:-1])
			slope[3,cp],bInt[3,cp]=np.polyfit(ydataNDVIAnom[:-1],cropYield[cp,1:],1)
			yfit=slope[3,cp]*ydataNDVIAnom[:-1]+bInt[3,cp]

		data2017[3,cp]=ydataNDVIAnom[-1]
	
		if makePlots:
			plt.clf()
			fig, ax2 = plt.subplots()
			ax1 = ax2.twinx()
			
			#if country=='Ethiopia':
			#	ax2.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,.4])
			#	if cp==0:
			#		ax1.set_ylim([6000,7300])
			#		#ax1.set_yticks([6000,6200,6400,6600,6800,7000,7200])
			#	if cp==2:
			#		ax1.set_ylim([3500,4450])
			#		ax2.set_ylim([0.06,0.40])
			#if country=='Tunisia':
			#	if cp==1:
			#		ax1.set_ylim([50,350])
			#		ax2.set_ylim([0.1,0.25])
			#	if cp==0:
			#		ax1.set_ylim([700,1520])
			#		ax2.set_ylim([0.11,0.25])
			#if country=='Morocco':
			#	ax1.set_ylim([2100,8300])
			#	ax2.set_ylim([0.48,0.71])
			ax1.set_ylim([-1*np.ma.amax(abs(ydata))*1.05,np.ma.amax(abs(ydata))*1.2])
			ax2.set_ylim([-1*np.ma.amax(abs(ydataNDVIAnom))*1.05,np.ma.amax(abs(ydataNDVIAnom))*1.2])
	
			ax2.set_ylabel('NDVI Monthly Average',color='b')
			#ax1.plot(x,ydata,'-*g')
			label=crop[cp]+' Production'
			ax1.bar(x,ydata,bar_width,color='g',label=label)
			ax1.tick_params(axis='y',colors='g')
			ax2.tick_params(axis='y',colors='b')
			#ax2.plot(xNDVI,ydataNDVIAnom,'-*b')
			ax2.bar(xNDVI,ydataNDVIAnom,bar_width,color='b',label='Max NDVI')
	
			ax1.set_ylabel('Production Anomaly, Gigatonnes',color='g')
			plt.title(country+': '+corrMonthName+' NDVI Anom and '+crop[cp]+' Production, Corr='+str(round(Corr[3,cp],2)))
			ax2.grid(True)
			ax1.set_xticks(range(2013,2013+nyears))
			ax1.legend(loc='upper right')
			ax2.legend(loc='upper left')
			plt.savefig(wdfigs+country+'/ndvi_anom_'+crop[cp]+'_'+country+'.jpg',dpi=700)
	
	
	###########################################
	# Monthly EVI Anom and Crop Yield
	###########################################
	
	ydataEVIAnom=np.zeros(shape=(nyears))
	if twoSeasons!='no':
		for y in range(nyears):
			ydatatmp1=np.ma.amax(eviAnom[y,corrMonth1-1:corrMonth1+2])
			if country=='Rwanda':
				if y!=0:
					ydatatmp2=np.ma.amax(eviAnom[y-1,corrMonth2-1:corrMonth2+2])
				else:
					ydataEVIAnom[y]=ydatatmp1
			else:
				ydatatmp2=np.ma.amax(eviAnom[y,corrMonth2-1:corrMonth2+2])
			ydataEVIAnom[y]=np.ma.mean([ydatatmp1,ydatatmp2])
	else:
		for y in range(nyears):
			ydataEVIAnom[y]=eviAnom[y,maxMonth[y]]
			if SeasonOverYear:
				ydataEVIAnom[y]=eviAnom[maxMonthWYears[y,0],maxMonthWYears[y,1]]
	if corrMonth!='Max':
		ydataEVIAnom=eviAnom[:,int(corrMonth)]
	
	for cp in range(len(crop)):
		plt.clf()
		fig, ax2 = plt.subplots()
		ax1 = ax2.twinx()
		ax2.grid(True)
	
		ydata=cropYieldAnom[cp,:]
		x=np.arange(2013+.14,2013+.14+nyears)
		xEVI=np.arange(2013-.14,2013-.14+nyears)
		Corr[4,cp]=corr(cropYieldAnom[cp,:],ydataEVIAnom)
		slope[4,cp],bInt[4,cp]=np.polyfit(ydataEVIAnom[:],cropYield[cp,:],1)
		yfit=slope[4,cp]*ydataEVIAnom[:]+bInt[4,cp]
	
		if corrYear=='after':
			ydata=cropYieldAnom[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[4,cp]=corr(cropYieldAnom[cp,1:],ydataEVIAnom[:-1])
			slope[4,cp],bInt[4,cp]=np.polyfit(ydataEVIAnom[:-1],cropYield[cp,1:],1)
			yfit=slope[4,cp]*ydataEVIAnom[:-1]+bInt[4,cp]
	
		data2017[4,cp]=ydataEVIAnom[-1]
	
		if makePlots:
			ax1.set_ylim([-1*np.ma.amax(abs(ydata))*1.05,np.ma.amax(abs(ydata))*1.2])
			ax2.set_ylim([-1*np.ma.amax(abs(ydataEVIAnom))*1.05,np.ma.amax(abs(ydataEVIAnom))*1.2])
			#ax1.set_ylim([np.amin(ydata)*1.1,np.ma.amax(ydata)*1.1])
			#ax2.set_ylim([np.amin(ydataEVIAnom)*1.1,np.ma.amax(ydataEVIAnom)*1.1])
	
			#if country=='Ethiopia':
			#	ax1.set_yticks([6000,6200,6400,6600,6800,7000,7200])
			#if country=='Ethiopia':
			#	Corr=corr(cropYieldAnom[cp,1:],eviAnom[0,:-1,7])
			#if country=='Morocco':
			#	x=np.arange(2013.7,2018.7)
			#	ydata=cropYieldAnom[cp,:]
			#if country=='Tunisia':
			#	x=np.arange(2013.7,2018.7)
			#	ydata=cropYieldAnom[cp,:]
			
			
			ax2.bar(xEVI,ydataEVIAnom,bar_width,color='b',label='Max EVI')
			#ax2.plot(xtime,eviAnomPlot,'b*-')
			label=crop[cp]+' Production'
			ax1.bar(x,ydata,bar_width,color='g',label=label)
			#ax1.plot(x,ydata,'-*g')
	
			ax1.set_xticks(range(2013,2013+nyears))
			ax1.legend(loc='upper right')
			ax2.legend(loc='upper left')
	
			ax2.set_ylabel('EVI Monthly Average',color='b')
			ax1.set_ylabel('Production Anomaly, Gigatonnes',color='g')
			ax1.tick_params(axis='y',colors='g')
			ax2.tick_params(axis='y',colors='b')
	
			plt.title(country+': '+corrMonthName+' EVI Anom and '+crop[cp]+' Production, Corr='+str(round(Corr[4,cp],2)))
			plt.savefig(wdfigs+country+'/evi_anom_'+crop[cp]+'_'+country+'.jpg',dpi=700)
	
	###########################################
	# Monthly NDWI Anom and Crop Yield
	###########################################
	
	ydataNDWIAnom=np.zeros(shape=(nyears))
	if twoSeasons!='no':
		for y in range(nyears):
			ydatatmp1=np.ma.amax(ndwiAnom[y,corrMonth1-1:corrMonth1+2])
			if country=='Rwanda':
				if y!=0:
					ydatatmp2=np.ma.amax(ndwiAnom[y-1,corrMonth2-1:corrMonth2+2])
				else:
					ydataNDWIAnom[y]=ydatatmp1
			else:
				ydatatmp2=np.ma.amax(ndwiAnom[y,corrMonth2-1:corrMonth2+2])
			ydataNDWIAnom[y]=np.ma.mean([ydatatmp1,ydatatmp2])
	else:
		for y in range(nyears):
			ydataNDWIAnom[y]=ndwiAnom[y,maxMonth[y]]
			if SeasonOverYear:
				ydataNDWIAnom[y]=ndwiAnom[maxMonthWYears[y,0],maxMonthWYears[y,1]]

	if corrMonth!='Max':
		ydataNDWIAnom=ndwiAnom[:,int(corrMonth)]
		corrMonthName=monthName[int(corrMonth)]
	
	for cp in range(len(crop)):
		plt.clf()
		fig, ax2 = plt.subplots()
		ax1 = ax2.twinx()
	
		ydata=cropYieldAnom[cp,:]
		x=np.arange(2013+.14,2013+.14+nyears)
		xNDWI=np.arange(2013-.14,2013-.14+nyears)
		Corr[5,cp]=corr(cropYieldAnom[cp,:],ydataNDWIAnom)
		slope[5,cp],bInt[5,cp]=np.polyfit(ydataNDWIAnom[:],cropYield[cp,:],1)
		yfit=slope[5,cp]*ydataNDWIAnom[:]+bInt[5,cp]
		corrMonthName=corrMonth
	
		if corrYear=='after':
			ydata=cropYieldAnom[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[5,cp]=corr(cropYieldAnom[cp,1:],ydataNDWIAnom[:-1])
			slope[5,cp],bInt[5,cp]=np.polyfit(ydataNDWIAnom[:-1],cropYield[cp,1:],1)
			yfit=slope[5,cp]*ydataNDWIAnom[:-1]+bInt[5,cp]

		data2017[5,cp]=ydataNDWIAnom[-1]
	
		if makePlots:
			ax1.set_ylim([-1*np.ma.amax(abs(ydata))*1.05,np.ma.amax(abs(ydata))*1.2])
			ax2.set_ylim([-1*np.ma.amax(abs(ydataNDWIAnom))*1.05,np.ma.amax(abs(ydataNDWIAnom))*1.2])
			#ax1.set_ylim([np.amin(ydata)*1.1,np.ma.amax(ydata)*1.1])
			#ax2.set_ylim([np.amin(ydataNDWIAnom)*1.1,np.ma.amax(ydataNDWIAnom)*1.1])
			
			ax2.bar(xNDWI,ydataNDWIAnom,bar_width,color='b',label='Max NDWI')
			label=crop[cp]+' Production'
			ax1.bar(x,ydata,bar_width,color='g',label=label)
	
			ax1.set_xticks(range(2013,2013+nyears))
			ax1.legend(loc='upper right')
			ax2.legend(loc='upper left')
	
			ax2.set_ylabel('Absolute Value of NDWI Monthly Average',color='b')
			ax1.set_ylabel('Production Anomaly, Gigatonnes',color='g')
			ax1.tick_params(axis='y',colors='g')
			ax2.tick_params(axis='y',colors='b')
	
			plt.title(country+': '+corrMonthName+' NDWI Anom and '+crop[cp]+' Production, Corr='+str(round(Corr[5,cp],2)))
			ax2.grid(True)
			plt.savefig(wdfigs+country+'/ndwi_anom_'+crop[cp]+'_'+country+'.jpg',dpi=700)

	variables=['ndviAvg','eviAvg','ndwiAvg','ndviAnom','eviAnom','ndwiAnom']
	
	CorrMask=np.zeros(shape=(Corr.shape),dtype=int)
	for var in range(6):
		for cp in range(len(crop)):
			if np.isnan(Corr[var,cp])==True:
				CorrMask[var,cp]=1
	Corr=np.ma.masked_array(Corr,CorrMask)
	maxCorr=np.amax(Corr)

	whereMaxCorrX=np.where(Corr==maxCorr)[0][0]
	whereMaxCorrY=np.where(Corr==maxCorr)[1][0]

	countryNum=int(countryNum)
	slopesAll[countryNum]=slope[whereMaxCorrX,whereMaxCorrY]
	bIntAll[countryNum]=bInt[whereMaxCorrX,whereMaxCorrY]

	data=Counter(maxMonth)
	common=data.most_common(1)[0][0]
	maxMonthAll[countryNum]=common
	seasonHight=int(maxMonthAll[countryNum])

	print round(maxCorr,2)
	print crop[whereMaxCorrY], variables[whereMaxCorrX]
	print round(maxMonthAll[countryNum],0)
	
	if maxCorr<.75:
		print 'low corr'

	#if twoSeasons=='no':
	#	fwrite.write(str(countryNum)+','+country+','+str(seasonHight)+'\n')
	#else:
	#	fwrite.write(str(countryNum)+','+country+','+str(corrMonth1)+'/'+str(int(corrMonth2))+'\n')

	#################################### Make Predictions ####################################
	ydataForPred=np.zeros(shape=(6))
	if SeasonOverYear:
		ydataForPred[0]=np.ma.amax([ndviAvg[-1,9:],ndviAvg2018])
		ydataForPred[1]=np.ma.amax([eviAvg[-1,9:],eviAvg2018])
		ydataForPred[2]=np.ma.amin([ndwiAvg[-1,9:],ndwiAvg2018])

		ydataForPred[3]=np.ma.amax([ndviAnom[-1,9:],ndviAnom2018])
		ydataForPred[4]=np.ma.amax([eviAnom[-1,9:],eviAnom2018])
		ydataForPred[5]=np.ma.amax([ndwiAnom[-1,9:],ndwiAnom2018])

	else:
		ydataForPred[0]=np.ma.amax([ndviAvg2018])
		ydataForPred[1]=np.ma.amax([eviAvg2018])
		ydataForPred[2]=np.ma.amin([ndwiAvg2018])

		ydataForPred[3]=np.ma.amax([ndviAnom2018])
		ydataForPred[4]=np.ma.amax([eviAnom2018])
		ydataForPred[5]=np.ma.amax([ndwiAnom2018])

	yieldPred=slope[whereMaxCorrX,whereMaxCorrY]*ydataForPred[whereMaxCorrX]+bInt[whereMaxCorrX,whereMaxCorrY]
	predAnom=yieldPred-meanYield

	stdDevYield=stdDev(cropYield[whereMaxCorrY])
	predFromStdDev=predAnom/stdDevYield


	##########################################################################################
	# Plot the Prediction
	##########################################################################################
	cp=whereMaxCorrY
	if variables[whereMaxCorrX]=='ndviAvg':
		xdata=np.ma.compressed(ndviAvg)
		xdata2018=np.ma.compressed(ndviAvg2018)
	elif variables[whereMaxCorrX]=='eviAvg':
		xdata=np.ma.compressed(eviAvg)
		xdata2018=np.ma.compressed(eviAvg2018)
	elif variables[whereMaxCorrX]=='ndwiAvg':
		xdata=np.ma.compressed(ndwiAvg)
		xdata2018=np.ma.compressed(ndwiAvg2018)

	elif variables[whereMaxCorrX]=='ndviAnom':
		xdata=np.ma.compressed(ydataNDVIAnom)
		xdata2018=np.ma.compressed(ndviAnom2018)
	elif variables[whereMaxCorrX]=='eviAnom':
		xdata=np.ma.compressed(ydataEVIAnom)
		xdata2018=np.ma.compressed(eviAnom2018)
	elif variables[whereMaxCorrX]=='ndwiAnom':
		xdata=np.ma.compressed(ydataNDWIAnom)
		xdata2018=np.ma.compressed(ndwiAnom2018)

	ydata=cropYield[cp]
	satellitePlot=np.zeros(shape=(xdata.shape[0]+ndviAvg2018.shape[0]))
	satellitePlot[:xdata.shape[0]]=xdata
	satellitePlot[xdata.shape[0]:]=xdata2018
	
	if variables[whereMaxCorrX][-3:]=='Avg':

		x=np.arange(2013.+(harvestMonth/12),2013.+(harvestMonth/12)+nyears)

		xTimeSmall=np.zeros(shape=(ndviAvg2018.shape))
		for m in range(len(xTimeSmall)):
			xTimeSmall[m]=2018+(m+.5)/12

		xtimeNew=np.zeros(shape=(xtime.shape[0]+ndviAvg2018.shape[0]))
		xtimeNew[:xtime.shape[0]]=xtime
		xtimeNew[xtime.shape[0]:]=xTimeSmall

		plt.clf()
		fig, ax2 = plt.subplots()
		ax1 = ax2.twinx()
		ax2.grid(True)
	
		label=crop[cp]+' production'
		ax2.bar(x,ydata,bar_width,color='g',label=label)
		label='predicted production'
		ax2.bar(x[-1]+1,yieldPred,bar_width,color='m',label=label)
		ax2.legend(loc='upper right')
		ax2.tick_params(axis='y',colors='g')
		ax2.set_ylabel(crop[cp]+' production, gigatonnes',color='g')
	
		ax2.set_ylim([np.ma.amin(ydata)*.06,np.ma.amax(ydata)*1.1])
		ax1.set_ylim([np.ma.amin(satellitePlot)*.9,np.ma.amax(satellitePlot)*1.15])
	
		ax1.plot(xtimeNew,satellitePlot,'b*-',label='monthly ndvi')
		ax1.legend(loc='upper left')

		ax1.set_ylabel('NDVI Monthly Average',color='b')
		ax1.tick_params(axis='y',colors='b')
	
		plt.title(country+': NDVI Monthly Average and '+crop[cp]+' Production, Corr='+str(round(Corr[whereMaxCorrX,cp],2)))
		plt.savefig(wdfigs+country+'/pred_monthly_'+variables[whereMaxCorrX]+'_avg_with_'+crop[cp]+'.jpg',dpi=700)
		print 'made Plot'
		exit()


fwrite.close()

#plt.clf()
#plt.plot(ydataNDWI,cropYield[2,:],'b*')
#plt.plot(ydataNDWI,yfit,'g-')
#plt.xlabel('NDWI')
#plt.ylabel('Wheat Yield')
#plt.title('Tunisia: NDWI against Wheat Production, Corr='+str(round(Corr[2,2],2)))
#plt.savefig(wdfigs+'other/Tunisia_NDWI_wheat_corr')

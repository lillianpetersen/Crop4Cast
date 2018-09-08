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

def get_second_highest(a):
	hi = mid = 0
	for x in a:
		if x > hi:
			mid = hi
			hi = x
		elif x < hi and x > mid:
			lo = mid
			mid = x
	return mid

def get_third_highest(a):
	hi = mid = third = 0
	for x in a:
		if x > hi:
			third = mid
			mid = hi
			hi = x
		elif x < hi and x > mid:
			third = mid
			mid = x
		elif x < mid and x > third:
			third = x
	return third

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

currentMonth=datetime.datetime.now().strftime("%Y-%m-%d").split('-')[1]
fwrite=open(wddata+'max_ndviMonths.csv','w')
fmap=open(wddata+'crop_predictions_2018'+currentMonth+'.csv','w')

makePlots=False
makeFinalPlots=True
MakePredictions=True
nyears=5
#nyears=4

CropPrediction=np.zeros(shape=(48))
slopesAll=np.zeros(shape=(48))
bIntAll=np.zeros(shape=(48))
maxMonthAll=np.zeros(shape=(48))
badCorrCountries=[]

countryList=[]
cropAll=[]
indexAll=[]
CorrsAll=np.zeros(shape=(48))
predictionsAll=np.zeros(shape=(48))
countryCounter=-1
harvestMonthAll=np.zeros(shape=(48),dtype=int)
harvestMonthAllName=[]

for icountry in range(47):
	countryNum=str(icountry+1)
	#countryNum='26'
	if countryNum=='26' or countryNum=='29' or countryNum=='22': # South Sudan or Gabon
		continue
	#if countryNum=='31':
	#	continue
	
	SeasonOverYear=False

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
			harvestMonth=float(tmp[7])
			seasons=tmp[8][:-1].split('/')
			twoSeasons=seasons[0]
			if twoSeasons!='no':
				corrMonth1=int(seasons[1])
				corrMonth2=int(seasons[2])
			break

	########### find countries with the right growing season ###########
	currentMonth=int(datetime.datetime.now().strftime("%Y-%m-%d").split('-')[1])
	Good=False
	fseason=open(wddata+'max_ndviMonths_final.csv','r')
	for line in fseason:
		tmp=line.split(',')
		if tmp[0]==str(countryNum):
			country=tmp[1]
			sName=country

			corrMonth=tmp[2][:-1]
			if len(corrMonth)>2:
				months=corrMonth.split('/')
				month1=corrMonth[0]
				month2=corrMonth[1]
				corrMonth=month1
			corrMonth=int(corrMonth)

			if int(corrMonth)==currentMonth-1 or int(corrMonth)==currentMonth-2 or int(corrMonth)==currentMonth-3:
				print '\nRunning',country
				Good=True
				break
			else:
				print country, 'has other season'
				break
	if Good==False:
		continue
	####################################################################

	if MakePredictions:
		try:
			ndviAnom2018=np.load(wdvars+country+'/2018/ndviAnom.npy')
			eviAnom2018=np.load(wdvars+country+'/2018/eviAnom.npy')
			ndwiAnom2018=np.load(wdvars+country+'/2018/ndwiAnom.npy')
	
			ndviAvg2018=np.load(wdvars+country+'/2018/ndviAvg.npy')
			eviAvg2018=np.load(wdvars+country+'/2018/eviAvg.npy')
			ndwiAvg2018Normal=np.load(wdvars+country+'/2018/ndwiAvg.npy')
		except:
			if corrYear=='same':
				continue

		if corrYear=='after':
			ndviAvg2018=ndviAvg[-1]
			eviAvg2018=eviAvg[-1]
			ndwiAvg2018Normal=ndwiAvg[-1]
	
			ndviAnom2018=ndviAnom[-1]
			eviAnom2018=eviAnom[-1]
			ndwiAnom2018=ndwiAnom[-1]
	
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
		elif crop[cp]=='Green_Coffee':
			crop[cp]='Coffee'
	
	### Find Crop Yield Anomaly ###
	cropYieldAnom=np.zeros(shape=(cropYield.shape))
	meanYield=np.zeros(shape=(len(crop)))
	for cp in range(len(crop)):
		meanYield[cp]=np.mean(cropYield[cp,:])
		for y in range(nyears):
			cropYieldAnom[cp,y]=cropYield[cp,y]-meanYield[cp]
	
	if np.ma.amax(cropYieldAnom)==0. and np.amin(cropYieldAnom)==0:
		cropYieldAnom[:]=1
	
	########### load variables ###########
	ndviAnom=np.load(wdvars+country+'/ndviAnom.npy')
	eviAnom=np.load(wdvars+country+'/eviAnom.npy')
	ndwiAnom=np.load(wdvars+country+'/ndwiAnom.npy')
	
	ndviAvg=np.load(wdvars+country+'/ndviAvg.npy')
	eviAvg=np.load(wdvars+country+'/eviAvg.npy')
	ndwiAvg=np.load(wdvars+country+'/ndwiAvg.npy')
	######################################
	countryList.append(country)
	countryCounter+=1

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
			if math.isnan(ndviAvg[y,m])==True or ndviAvg[y,m]<0:
				MaskAvg[y,m]=1
			if math.isnan(ndviAnom[y,m])==True or ndviAvg[y,m]<0:
				MaskAnom[y,m]=1
	
	ndviAvg=np.ma.masked_array(ndviAvg,MaskAvg)
	eviAvg=np.ma.masked_array(eviAvg,MaskAvg)
	ndwiAvgNormal=np.ma.masked_array(ndwiAvg,MaskAvg)
	ndviAnom=np.ma.masked_array(ndviAnom,MaskAnom)
	eviAnom=np.ma.masked_array(eviAnom,MaskAnom)
	ndwiAnom=np.ma.masked_array(ndwiAnom,MaskAnom)
	xtime=np.ma.masked_array(xtime,MaskAvg)

	ndwiAnom=abs(ndwiAnom)

	ndwiAvg=np.zeros(shape=(ndwiAvgNormal.shape))
	mean=np.mean(ndwiAvgNormal)
	for y in range(nyears):
		for m in range(12):
			ndwiAvg[y,m]=(1./(ndwiAvgNormal[y,m]/mean))*mean 
	
	if MakePredictions:
		ndwiAnom2018=abs(ndwiAnom2018)
		if corrYear=='same':
			ndwiAvg2018=np.zeros(shape=(ndwiAvg2018Normal.shape))
			for m in range(3):
				ndwiAvg2018[m]=(1./(ndwiAvg2018Normal[m]/mean))*mean 
	
		if corrYear=='after':
			ndwiAvg2018=np.zeros(shape=(ndwiAvg2018Normal.shape))
			for m in range(12):
				ndwiAvg2018[m]=(1./(ndwiAvg2018Normal[m]/mean))*mean 
	
	ndwiAvg=np.ma.masked_array(ndwiAvg,MaskAvg)
	
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
					ydatatmp2=ydatatmp1
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
			maxtmp=np.ma.amax(ndviAvg[0,:6])
			maxMonthWYears[0,:]=np.ma.where(ndviAvg[:,:]==maxtmp)[0][0],np.ma.where(ndviAvg[:,:]==maxtmp)[1][0]
			for y in range(1,nyears):
				maxtmp1=np.ma.amax(ndviAvg[y-1,6:])
				maxtmp2=np.ma.amax(ndviAvg[y,:6])
				maxtmp=np.ma.amax([maxtmp1,maxtmp2])
				maxMonthWYears[y,:]=np.ma.where(ndviAvg[:,:]==maxtmp)[0][0],np.ma.where(ndviAvg[:,:]==maxtmp)[1][0]

		if SeasonOverYear:
			for y in range(nyears):
				ydataNDVI[y]=ndviAvg[maxMonthWYears[y,0],maxMonthWYears[y,1]]

	## Mask the index ##
	wherenan=np.where(np.isnan(ydataNDVI)==True)
	masktmp=np.zeros(shape=5)
	masktmp[wherenan]=1
	if np.sum(masktmp)>0:
		for y in range(nyears):
			ydataNDVI[y]=np.amax(ndviAvg[y])
	#ydataNDVI=np.ma.masked_array(ydataNDVI,masktmp)
	####################

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
			ax2.grid(True)
	
			label=crop[cp]+' Production'
			ax2.bar(x,ydata,bar_width,color='g',label=label)
			ax2.legend(loc='upper right')
			ax2.tick_params(axis='y',colors='g')
			ax2.set_ylabel(crop[cp]+' Production, Gigatonnes',color='g')
	
			#if country=='ethiopia':
			#	ax2.set_ylim([6000,7300])
			#if country=='morocco':
			#	ax2.set_yticks([0,1000,2000,3000,4000,5000,6000,7000,8000])
			ax2.set_ylim([np.ma.amin(ydata)*.06,np.ma.amax(ydata)*1.15])
			ax1.set_ylim([np.ma.amin(ndviAvgPlot)*.9,np.ma.amax(ydataNDVI)*1.1])
	
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
			
			ax1.plot(xtime,ndviAvgPlot,'b*-',label='monthly ndvi')
			ax1.legend(loc='upper left')
			#if country=='ethiopia':
			#	ax1.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
			#if country=='morocco':
			#	ax1.set_yticks([0.25,0.3,0.35,.4,.45,.5,.55,.6,.65])
			#else:
			#	ax1.set_ylim([0,np.ma.amax(ndviavgplot)*1.2])
			ax1.set_ylabel('ndvi monthly average',color='b')
			ax1.tick_params(axis='y',colors='b')
	
			plt.title(country+': NDVI Monthly Average and '+crop[cp]+' Production, Corr='+str(round(Corr[0,cp],2)))
			plt.savefig(wdfigs+country+'/monthly_ndvi_avg_with_'+crop[cp]+'.pdf',dpi=700)

	
	
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
			plt.savefig(wdfigs+country+'/ndvi_avg_'+crop[cp]+'_'+country+'.pdf',dpi=700)
	
	
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
					ydatatmp2=ydatatmp1
			else:
				ydatatmp2=np.ma.amax(eviAvg[y,corrMonth2-1:corrMonth2+2])
			ydataEVI[y]=np.ma.mean([ydatatmp1,ydatatmp2])
	else:
		for y in range(nyears):
			ydataEVI[y]=eviAvg[y,maxMonth[y]]
		#	ydataEVI[y]=np.ma.amax(eviAvg[y,:])
		#	maxMonth[y]=np.ma.where(eviAvg[y,:]==np.ma.amax(eviAvg[y,:]))[0][0]
		#if np.any(maxMonth==10) or np.any(maxMonth==11) or np.any(maxMonth==0) or np.any(maxMonth==1):
		#	SeasonOverYear=True
		#	maxMonthWYears=np.zeros(shape=(nyears,2),dtype=int)
		#	maxtmp=np.ma.amax(eviAvg[0,:6])
		#	maxMonthWYears[0,:]=np.ma.where(eviAvg[:,:]==maxtmp)[0][0],np.ma.where(eviAvg[:,:]==maxtmp)[1][0]
		#	for y in range(1,nyears):
		#		maxtmp1=np.ma.amax(eviAvg[y-1,6:])
		#		maxtmp2=np.ma.amax(eviAvg[y,:6])
		#		maxtmp=np.ma.amax([maxtmp1,maxtmp2])
		#		maxMonthWYears[y,:]=np.ma.where(eviAvg[:,:]==maxtmp)[0][0],np.ma.where(eviAvg[:,:]==maxtmp)[1][0]
		if SeasonOverYear:
			for y in range(nyears):
				ydataNDVI[y]=ndviAvg[maxMonthWYears[y,0],maxMonthWYears[y,1]]

	corrMonthName=corrMonth
	if corrMonth!='Max':
		ydataEVI=eviAvg[:,int(corrMonth)]
		corrMonthName=monthName[int(corrMonth)]

	## Mask the index ##
	wherenan=np.where(np.isnan(ydataEVI)==True)
	masktmp=np.zeros(shape=5)
	masktmp[wherenan]=1
	if np.sum(masktmp)>0:
		for y in range(nyears):
			ydataEVI[y]=np.amax(eviAvg[y])
	#ydataEVI=np.ma.masked_array(ydataEVI,masktmp)
	####################
	
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
			#ax1.set_ylim([np.ma.amin(ydata)*0.9,np.ma.amax(ydata)*1.05])
			ax1.set_ylim([0,np.ma.amax(ydata)*1.15])
			ax2.set_ylim([np.ma.amin(ydataEVI)*0.99,np.ma.amax(ydataEVI)*1.03])
	
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
			plt.savefig(wdfigs+country+'/evi_avg_'+crop[cp]+'_'+country+'.pdf',dpi=700)
	
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
					ydatatmp2=ydatatmp1
			else:
				ydatatmp2=np.ma.amax(ndwiAvg[y,corrMonth2-1:corrMonth2+2])
			ydataNDWI[y]=np.ma.mean([ydatatmp1,ydatatmp2])
	else:
		for y in range(nyears):
			ydataNDWI[y]=ndwiAvg[y,maxMonth[y]]
			#ydataNDWI[y]=np.ma.amax(ndwiAvg[y,:])
		if SeasonOverYear:
			for y in range(nyears):
				ydataNDWI[y]=ndwiAvg[maxMonthWYears[y,0],maxMonthWYears[y,1]]

	corrMonthName=corrMonth

	if corrMonth!='Max' and twoSeasons=='no':
		ydataNDWI=np.zeros(shape=(nyears))
		ydataNDWI=np.zeros(shape=(nyears))
		for y in range(nyears):
			ydataNDWI[y]=ndwiAvg[y,int(corrMonth)]
		mean=np.mean(ydataNDWI[y])
		for y in range(nyears):
			ydataNDWI[y]=(1./(ydataNDWI[y]/mean))*mean 
		corrMonthName=monthName[int(corrMonth)]

	## Mask the index ##
	wherenan=np.where(np.isnan(ydataNDWI)==True)
	masktmp=np.zeros(shape=5)
	masktmp[wherenan]=1
	if np.sum(masktmp)>0:
		for y in range(nyears):
			ydataNDWI[y]=np.amax(ndwiAvg[y])
	#ydataNDWI=np.ma.masked_array(ydataNDWI,masktmp)
	####################
	
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
	
	for cp in range(len(crop)):
		if makePlots:

			if np.mean(ydataNDWI)<0:
				ydataNDWI2=abs(ydataNDWI)

				mean=np.mean(ydataNDWI2)
				for y in range(nyears):
					ydataNDWI2[y]=(1./(ydataNDWI2[y]/mean))*mean 


			ax1.set_ylim([np.ma.amin(ydata)*0.9,np.ma.amax(ydata)*1.05])
			ax2.set_ylim([np.ma.amin(ydataNDWI2)*0.9,np.ma.amax(ydataNDWI2)*1.05])
			
			ax2.bar(xNDWI,ydataNDWI2,bar_width,color='b',label='Max NDWI')
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
			plt.savefig(wdfigs+country+'/ndwi_avg_'+crop[cp]+'_'+country+'.pdf',dpi=700)
	
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
						ydatatmp2=ydatatmp1
				else:
					ydatatmp2=np.ma.amax(ndviAnom[y,corrMonth2-1:corrMonth2+2])
				ydataNDVIAnom[y]=np.ma.mean([ydatatmp1,ydatatmp2])
		else:
			for y in range(nyears):
				ydataNDVIAnom[y]=ndviAnom[y,maxMonth[y]]
				if SeasonOverYear:
					ydataNDVIAnom[y]=ndviAnom[maxMonthWYears[y,0],maxMonthWYears[y,1]]

		## Mask the index ##
		wherenan=np.where(np.isnan(ydataNDVIAnom)==True)
		masktmp=np.zeros(shape=5)
		masktmp[wherenan]=1
		if np.sum(masktmp)>0:
			for y in range(nyears):
				ydataNDVIAnom[y]=np.amax(ndviAnom[y])
		#ydataNDVIAnom=np.ma.masked_array(ydataNDVIAnom,masktmp)
		####################

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
			plt.savefig(wdfigs+country+'/ndvi_anom_'+crop[cp]+'_'+country+'.pdf',dpi=700)
	
	
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
					ydatatmp2=ydatatmp1
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

	## Mask the index ##
	wherenan=np.where(np.isnan(ydataEVIAnom)==True)
	masktmp=np.zeros(shape=5)
	masktmp[wherenan]=1
	if np.sum(masktmp)>0:
		for y in range(nyears):
			ydataEVIAnom[y]=np.amax(eviAnom[y])
	#ydataEVIAnom=np.ma.masked_array(ydataEVIAnom,masktmp)
	####################
	
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
			plt.savefig(wdfigs+country+'/evi_anom_'+crop[cp]+'_'+country+'.pdf',dpi=700)
	
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
					ydatatmp2=ydatatmp1
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

	## Mask the index ##
	wherenan=np.where(np.isnan(ydataNDWIAnom)==True)
	masktmp=np.zeros(shape=5)
	masktmp[wherenan]=1
	if np.sum(masktmp)>0:
		print 'np.sum(masktmp)>0'
		for y in range(nyears):
			ydataNDWIAnom[y]=np.amax(ndwiAnom[y])
	#ydataNDWIAnom=np.ma.masked_array(ydataNDWIAnom,masktmp)
	####################
	
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
			plt.savefig(wdfigs+country+'/ndwi_anom_'+crop[cp]+'_'+country+'.pdf',dpi=700)

	variablesTitle=['NDVI Avg','EVI Avg','NDWI Avg','NDVI Anom','EVI Anom','NDWI Anom']
	variables=['ndviAvg','eviAvg','ndwiAvg','ndviAnom','eviAnom','ndwiAnom']
	
	CorrMask=np.zeros(shape=(Corr.shape),dtype=int)
	for var in range(6):
		for cp in range(len(crop)):
			if np.isnan(Corr[var,cp])==True:
				CorrMask[var,cp]=1
	Corr=np.ma.masked_array(Corr,CorrMask)
	maxCorr=np.ma.amax(Corr)
	maxCorr2=get_second_highest(np.ma.compressed(Corr))
	#maxCorr3=get_third_highest(np.ma.compressed(Corr))

	whereMaxCorrX=np.where(Corr==maxCorr)[0][0]
	#whereMaxCorrX2=np.where(Corr==maxCorr2)[0][0]
	#whereMaxCorrX3=np.where(Corr==maxCorr3)[0][0]
	whereMaxCorrY=np.where(Corr==maxCorr)[1][0]
	#whereMaxCorrX=2
	#whereMaxCorrY=2

	countryNum=int(countryNum)
	slopesAll[countryNum]=slope[whereMaxCorrX,whereMaxCorrY]
	bIntAll[countryNum]=bInt[whereMaxCorrX,whereMaxCorrY]

	data=Counter(maxMonth)
	common=data.most_common(1)[0][0]
	maxMonthAll[countryNum]=common
	harvestMonthAll[countryCounter]=maxMonthAll[countryNum]+3
	seasonHight=int(maxMonthAll[countryNum])

	if harvestMonthAll[countryCounter]>11:
		harvestMonthAll[countryCounter]=harvestMonthAll[countryCounter]-12
	harvestMonthAll[0]=11

	print round(maxCorr,2)
	print crop[whereMaxCorrY], variables[whereMaxCorrX]
	print round(maxMonthAll[countryNum],0)

	CorrsAll[countryCounter]=maxCorr
	cropAll.append(crop[whereMaxCorrY])
	indexAll.append(variables[whereMaxCorrX])
	
	if maxCorr<.75:
		print 'low corr'

	if twoSeasons=='no':
		fwrite.write(str(countryNum)+','+country+','+str(seasonHight)+'\n')
	else:
		fwrite.write(str(countryNum)+','+country+','+str(corrMonth1)+'/'+str(int(corrMonth2))+'\n')

	if MakePredictions:
		#################################### Make Predictions ####################################
		ydataForPred=np.zeros(shape=(6))
		if countryNum==3 or countryNum==26 or countryNum==37 or countryNum==40 or countryNum==41 or countryNum==44 or countryNum==45:
			nmonths=6
		else:
			nmonths=6
		if corrYear=='same':
			ydataForPred[0]=np.ma.amax(ndviAvg2018)
			month2018=np.where(ndviAvg2018==ydataForPred[0])[0][0]
			ydataForPred[1]=eviAvg2018[month2018]
			ydataForPred[2]=ndwiAvg2018[month2018]

			ydataForPred[3]=ndviAnom2018[month2018]
			ydataForPred[4]=eviAnom2018[month2018]
			ydataForPred[5]=ndwiAnom2018[month2018]

			#ydatatmp1=np.ma.amax(ndviAvg[-1,9:])
			#ydatatmp2=np.ma.amax(ndviAvg2018)
			#ydataForPred[0]=np.ma.amax([ydatatmp1,ydatatmp2])
			#month20181=np.where([np.array(ndviAvg[-1,12-nmonths:]),ndviAvg2018]==ydataForPred[0])[0][0]
			#month20182=np.where([np.array(ndviAvg[-1,12-nmonths:]),ndviAvg2018]==ydataForPred[0])[1][0]

			#ydataForPred[1]=[eviAvg[-1,12-nmonths:],eviAvg2018][month20181][month20182]
			#ydataForPred[2]=[ndwiAvg[-1,12-nmonths:],ndwiAvg2018][month20181][month20182]

			#ydataForPred[3]=[ndviAnom[-1,12-nmonths:],ndviAnom2018][month20181][month20182]
			#ydataForPred[4]=[eviAnom[-1,12-nmonths:],eviAnom2018][month20181][month20182]
			#ydataForPred[5]=[ndwiAnom[-1,12-nmonths:],ndwiAnom2018][month20181][month20182]

		else:
			ydataForPred[0]=ydataNDVI[-1]
			ydataForPred[1]=ydataEVI[-1]
			ydataForPred[2]=ydataNDWI[-1]

			ydataForPred[3]=ydataNDVIAnom[-1]
			ydataForPred[4]=ydataEVIAnom[-1]
			ydataForPred[5]=ydataNDWIAnom[-1]
			if np.isnan(ydataForPred[5])==True:
				print 'ydataForPred[5]=nan'
				ydataForPred[5]=np.ma.mean([ndwiAnom2018[maxMonth[-1]-1],ndwiAnom2018[maxMonth[-1]+2]])

		if twoSeasons!='no':
			ydataForPred[0]=np.amax(ndviAvg2018[corrMonth-1:corrMonth+2])
			month2018=np.where(ndviAvg2018==ydataForPred[0])[0][0]

			ydataForPred[1]=eviAvg2018[month2018]
			ydataForPred[2]=ndwiAvg2018[month2018]

			ydataForPred[3]=ndviAnom2018[month2018]
			ydataForPred[4]=eviAnom2018[month2018]
			ydataForPred[5]=ndwiAnom2018[month2018]
			

		yieldPred=slope[whereMaxCorrX,whereMaxCorrY]*ydataForPred[whereMaxCorrX]+bInt[whereMaxCorrX,whereMaxCorrY]
		predAnom=yieldPred-meanYield[whereMaxCorrY]

		stdDevYield=stdDev(cropYield[whereMaxCorrY])
		predFromStdDev=predAnom/stdDevYield


		##########################################################################################
		# data for plot 
		##########################################################################################
		#if whereMaxCorrX>=2:
		#	whereMaxCorrX=whereMaxCorrX2
		#	print variables[whereMaxCorrX2]

		#if whereMaxCorrX>=2:
		#	whereMaxCorrX=whereMaxCorrX3
		#	print variables[whereMaxCorrX]

		bar_width=0.2
		cp=whereMaxCorrY
		if variables[whereMaxCorrX]=='ndviAvg':
			xdataMonthly=np.ma.compressed(ndviAvg)
			xdata=np.ma.compressed(ydataNDVI)
			xdata2018=np.ma.compressed(ndviAvg2018)
		elif variables[whereMaxCorrX]=='eviAvg':
			xdataMonthly=np.ma.compressed(eviAvg)
			xdata=np.ma.compressed(ydataEVI)
			xdata2018=np.ma.compressed(eviAvg2018)
		elif variables[whereMaxCorrX]=='ndwiAvg':
			xdataMonthly=np.ma.compressed(ndwiAvg)
			xdata=np.ma.compressed(ydataNDWI)
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

		harvestMonth=harvestMonth-.5

		if corrYear=='after' and harvestMonth<6:
			x=np.arange(2013.+(harvestMonth/12)+1,2013.+(harvestMonth/12)+nyears)
			ydata=cropYield[cp,1:]
			if variables[whereMaxCorrX][-1]=='m':
				ydata=cropYieldAnom[cp,1:]
		elif corrYear=='after' and harvestMonth>6:
			x=np.arange(2013.+(harvestMonth/12),2013.+(harvestMonth/12)+nyears-1)
			ydata=cropYield[cp,1:]
			if variables[whereMaxCorrX][-1]=='m':
				ydata=cropYieldAnom[cp,1:]
		else:
			ydata=cropYield[cp]
			x=np.arange(2013.+(harvestMonth/12),2013.+(harvestMonth/12)+nyears)
			if variables[whereMaxCorrX][-1]=='m':
				ydata=cropYieldAnom[cp]

		###########################################
		# Prediction Plots
		###########################################

		if makeFinalPlots:
			if variables[whereMaxCorrX]=='ndviAvg':
		
				if corrYear=='same':
					satellitePlot=np.zeros(shape=(xdataMonthly.shape[0]+ndviAvg2018.shape[0]))
					satellitePlot[:xdataMonthly.shape[0]]=xdataMonthly
					satellitePlot[xdataMonthly.shape[0]:]=xdata2018
				else:
					satellitePlot=xdataMonthly
		
				xTimeSmall=np.zeros(shape=(ndviAvg2018.shape))
				for m in range(len(xTimeSmall)):
					xTimeSmall[m]=2018+(m+.5)/12
		
				if corrYear=='same':
					xtimeNew=np.zeros(shape=(xtime.shape[0]+ndviAvg2018.shape[0]))
					xtimeNew[:xtime.shape[0]]=xtime
					xtimeNew[xtime.shape[0]:]=xTimeSmall
				else:
					xtimeNew=xtime
		
				plt.clf()
				fig, ax2 = plt.subplots()
				ax1 = ax2.twinx()
				ax2.grid(True)
			
				label=crop[cp]+' Production'
				ax2.bar(x,ydata,bar_width,color='g',label=label)
				label='Predicted Production'
				ax2.bar(x[-1]+1,yieldPred,bar_width,color='m',label=label)
				ax2.legend(loc='upper right')
				ax2.tick_params(axis='y',colors='g')
				ax2.set_ylabel(crop[cp]+' Production, Gigatonnes',color='g')
			
				ax2.set_ylim([np.ma.amin(ydata)*.96,np.ma.amax(ydata)*1.02])
				ax1.set_ylim([np.ma.amin(satellitePlot*100)*.9,np.ma.amax(satellitePlot*100)*1.15])
			
				ax1.plot(xtimeNew,satellitePlot*100,'b*-',label='Monthly NDVI')
				ax1.legend(loc='upper left')
		
				ax1.set_ylabel(variablesTitle[whereMaxCorrX]+' Monthly Average *100',color='b')
				ax1.tick_params(axis='y',colors='b')
			
				plt.title(country+': '+variablesTitle[whereMaxCorrX]+' and '+crop[cp]+' Prod, Pred='+str(round(predFromStdDev,2))+' Std Dev from Avg')
				plt.savefig(wdfigs+country+'/pred_monthly_'+variables[whereMaxCorrX]+'_avg_with_'+crop[cp]+'.pdf',dpi=700)
			
		###########################################################
		
			if variables[whereMaxCorrX]=='ndviAvg' or variables[whereMaxCorrX]=='eviAvg' or variables[whereMaxCorrX]=='ndwiAvg': # Bar Prediction Plot
				bar_width = 0.27
		
				if corrYear=='same':
					satellitePlot=np.zeros(shape=(xdata.shape[0]+1))
					satellitePlot[:xdata.shape[0]]=xdata
					satellitePlot[xdata.shape[0]:]=ydataForPred[whereMaxCorrX]
				else:
					satellitePlot=xdata
		
				if np.mean(satellitePlot)<0:
					satellitePlot=abs(satellitePlot)
		
					mean=np.mean(satellitePlot)
					for y in range(nyears):
						satellitePlot[y]=(1./(satellitePlot[y]/mean))*mean 
		
		
				if corrYear=='same':
					xNDVI=np.arange(2013-.14,2013+nyears+1-.14)
					x=np.arange(2013+.14,2013+nyears+.14)
				else:
					xNDVI=np.arange(2013-.14,2013+nyears-.14)
					x=np.arange(2013+.14,2013+nyears+.14)
		
				plt.clf()
				fig, ax2 = plt.subplots()
				ax1 = ax2.twinx()
				ax3 = ax2.twinx()
		
				if country=='Ethiopia':
					ax1.set_ylim([np.ma.amin(ydata)*.96,np.ma.amax(ydata)*1.035]) # for Ethiopia
					ax2.set_ylim([np.ma.amin(satellitePlot*100)*.65,np.ma.amax(satellitePlot*100)*1.15])
				#ax1.set_ylim([np.ma.amin(ydata)*.99,np.ma.amax(ydata)*1.005]) 
				#ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.6,np.ma.amax(satellitePlot*100)*1.15])
				elif country=='Swaziland':
					ax1.set_ylim([0,np.ma.amax(ydata)*1.39]) # for Swaziland
					ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.9,np.ma.amax(satellitePlot*100)*1.07])
				elif country=='Botswana':
					ax1.set_ylim([0,np.ma.amax(ydata)*1.15]) # for botswana
					ax2.set_ylim([np.ma.amin(ydataEVI*100)*0.99,np.ma.amax(ydataEVI*100)*1.03])
				elif country=='Zimbabwe':
					ax1.set_ylim([0,np.ma.amax(ydata)*1.15]) 
					ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.9,np.ma.amax(satellitePlot*100)*1.05])
				elif country=='Lesotho':
					ax1.set_ylim([0,np.ma.amax(ydata)*1.25]) 
					ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.9,np.ma.amax(satellitePlot*100)*1.05])
				elif country=='Malawi':
					ax1.set_ylim([np.ma.amin(ydata)*.9,np.ma.amax(ydata)*1.07]) 
					ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.99,np.ma.amax(satellitePlot*100)*1.01])
				elif country=='Morocco':
					ax1.set_ylim([np.ma.amin(ydata)*.85,np.ma.amax(ydata)*1.25]) 
					ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.90,np.ma.amax(satellitePlot*100)*1.15])
				else:
					ax1.set_ylim([np.ma.amin(ydata)*.9,np.ma.amax(ydata)*1.07]) 
					ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.9,np.ma.amax(satellitePlot*100)*1.07])
		
			
				if variablesTitle[whereMaxCorrX]=='NDWI Avg':
					ax2.set_ylabel('Max '+variablesTitle[whereMaxCorrX][:4]+' Month Average *100, flipped',color='b')
				else:
					ax2.set_ylabel('Max '+variablesTitle[whereMaxCorrX][:4]+' Month Average *100',color='b')
		
				label=crop[cp]+' Production'
				if corrYear=='after':
					ax1.bar(x[:-1],ydata,bar_width,color='g',label=label)
					label='Predicted production'
					ax1.bar(x[-1],yieldPred,bar_width,color='m',label=label)
				else:
					ax1.bar(x,ydata,bar_width,color='g',label=label)
					label='Predicted production'
					ax1.bar(x[-1]+1,yieldPred,bar_width,color='m',label=label)
				ax1.tick_params(axis='y',colors='g')
				ax2.tick_params(axis='y',colors='b')
				ax2.bar(xNDVI,satellitePlot*100,bar_width,color='b',label='Max '+variablesTitle[whereMaxCorrX])
			
				# Corr text
				if corrYear=='same':
					props = dict(boxstyle='round', facecolor='white', alpha=1)
					ax3.text(2014.8, .93, 'Corr='+str(round(maxCorr,2)),bbox=props)
				else:
					props = dict(boxstyle='round', facecolor='white', alpha=1)
					ax3.text(2014.4, .93, 'Corr='+str(round(maxCorr,2)),bbox=props)
		
				ax1.set_ylabel('Production, Gigatonnes',color='g')
				ax3.axis('off')
				plt.title(country+': '+variablesTitle[whereMaxCorrX][:4]+' and '+crop[cp]+' Prod, Pred='+str(round(predFromStdDev,2))+' Std Dev from Avg')
				ax2.grid(True)
				if corrYear=='after':
					ax1.set_xticks(range(2013,2013+nyears))
					ax2.set_xticks(range(2013,2013+nyears))
				else:
					ax1.set_xticks(range(2013,2013+nyears+1))
					ax2.set_xticks(range(2013,2013+nyears+1))
				ax1.legend(loc='upper right')
				ax2.legend(loc='upper left')
				#plt.savefig(wdfigs+country+'/'+country+'_pred_monthly_'+variables[whereMaxCorrX]+'_avg_with_'+crop[cp]+'.pdf',dpi=700)
				plt.savefig(wdfigs+'current_harvest'+'/'+country+'_pred_monthly_'+variables[whereMaxCorrX]+'_avg_with_'+crop[cp]+'.pdf',dpi=700)
		
		###########################################################
		
			if variables[whereMaxCorrX]=='ndviAnom' or variables[whereMaxCorrX]=='eviAnom' or variables[whereMaxCorrX]=='ndwiAnom': # Bar Prediction Plot
				bar_width = 0.27
		
				if corrYear=='same':
					satellite=np.zeros(shape=(xdata.shape[0]+1))
					satellite[:xdata.shape[0]]=xdata
					satellite[xdata.shape[0]:]=ydataForPred[whereMaxCorrX]
				else:
					satellite=xdata
		
				satellitePlot=np.zeros(shape=(satellite.shape))
				sMean=np.mean(satellite)
				for y in range(len(satellite)):
					satellitePlot[y]=satellite[y]-sMean
		
				if corrYear=='same':
					xNDVI=np.arange(2013-.14,2013+nyears+1-.14)
					x=np.arange(2013+.14,2013+nyears+.14)
				else:
					xNDVI=np.arange(2013-.14,2013+nyears-.14)
					x=np.arange(2013+.14,2013+nyears+.14)
		
				plt.clf()
				fig, ax2 = plt.subplots()
				ax1 = ax2.twinx()
				ax3 = ax2.twinx()
		
				ydataAbs1=np.ma.amax(abs(ydata))
				ydataAbs2=abs(predAnom)
				ydataAbs=np.ma.amax([ydataAbs1,ydataAbs2])
		
				ax1.set_ylim([ydataAbs*-1.2,ydataAbs*1.3]) 
				ax2.set_ylim([np.ma.amax(abs(satellitePlot*100))*-1.2,np.ma.amax(abs(satellitePlot*100))*1.3])
		
				ax2.set_ylabel(variablesTitle[whereMaxCorrX][:4]+' Anomaly *100',color='b')
		
				label=crop[cp]+' Production'
				if corrYear=='after':
					ax1.bar(x[:-1],ydata+1000,bar_width,bottom=-1000,color='g',label=label)
					label='Predicted Production'
					ax1.bar(x[-1],predAnom+1000,bar_width,bottom=-1000,color='m',label=label)
				else:
					ax1.bar(x,ydata+1000,bar_width,bottom=-1000,color='g',label=label)
					label='Predicted Production'
					ax1.bar(x[-1]+1,predAnom+100,bar_width,bottom=-100,color='m',label=label)
				ax1.tick_params(axis='y',colors='g')
				ax2.tick_params(axis='y',colors='b')
				ax2.bar(xNDVI,satellitePlot*100+100,bar_width,bottom=-100,color='b',label=variablesTitle[whereMaxCorrX])
			
				# Corr text
				props = dict(boxstyle='round', facecolor='white', alpha=1)
				if corrYear=='same':
					props = dict(boxstyle='round', facecolor='white', alpha=1)
					if country=='Rwanda':
						ax3.text(2014.8, .93, 'Corr='+str(round(maxCorr,3)),bbox=props)
					else:
						ax3.text(2014.8, .93, 'Corr='+str(round(maxCorr,2)),bbox=props)
				else:
					props = dict(boxstyle='round', facecolor='white', alpha=1)
					if country=='Rwanda':
						ax3.text(2014.4, .93, 'Corr='+str(round(maxCorr,3)),bbox=props)
					else:
						ax3.text(2014.4, .93, 'Corr='+str(round(maxCorr,2)),bbox=props)
		
				ax3.axis('off')
				ax1.set_ylabel('Production Anomaly, Gigatonnes',color='g')
				plt.title(country+': '+variablesTitle[whereMaxCorrX][:4]+' and '+crop[cp]+' Prod, Pred='+str(round(predFromStdDev,1))+' Std Dev from Avg')
				ax2.grid(True)
				if corrYear=='same':
					ax1.set_xticks(range(2013,2013+nyears+1))
				else:
					ax1.set_xticks(range(2013,2013+nyears))
				ax1.legend(loc='upper right')
				ax2.legend(loc='upper left')
				#plt.savefig(wdfigs+country+'/'+country+'_pred_monthly_'+variables[whereMaxCorrX]+'_avg_with_'+crop[cp]+'.pdf',dpi=700)
				plt.savefig(wdfigs+'current_harvest'+'/'+country+'_pred_monthly_'+variables[whereMaxCorrX]+'_avg_with_'+crop[cp]+'.pdf',dpi=700)
		
			if maxCorr>.75:
				print 'good corr'
			else:
				badCorrCountries.append(country)
		
		predictionsAll[countryCounter]=predFromStdDev

		###########################################
		# Write to file for interactive map
		###########################################

		if country=='DR Congo':
			country='Democratic Republic of the Congo'
		if country=='Central Africa Republic':
			country='Central African Republic'
		if country=='Tanzania':
			country='United Republic of Tanzania'
		if country=='Libya':
			country='Libyan Arab Jamahiriya'

	harvestMonthAllName.append(monthName[harvestMonthAll[countryCounter]])

	fmap.write(country+','+str(predictionsAll[countryCounter])+','+str(CorrsAll[countryCounter])+','+cropAll[countryCounter]+','+harvestMonthAllName[countryCounter]+','+indexAll[countryCounter]+'\n')

plt.clf()
n, bins, patches = plt.hist(CorrsAll, bins=10, range=(0,1),facecolor='blue',edgecolor='black', alpha=.9)
plt.title('Every African Country\'s Max Correlation to Crop Production')
plt.ylabel('Number of Countries')
plt.xlabel('Correlation')
plt.savefig(wdfigs+'three_countries/Africa_correlations_hist.pdf',dpi=700)

fwrite.close()
fmap.close()

#plt.clf()
#plt.plot(ydataNDWI,cropYield[2,:],'b*')
#plt.plot(ydataNDWI,yfit,'g-')
#plt.xlabel('NDWI')
#plt.ylabel('Wheat Yield')
#plt.title('Tunisia: NDWI against Wheat Production, Corr='+str(round(Corr[2,2],2)))
#plt.savefig(wdfigs+'other/Tunisia_NDWI_wheat_corr')

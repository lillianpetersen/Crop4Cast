import sys
import os
import matplotlib.pyplot as plt
#import descarteslabs as dl
import numpy as np
import math
from sys import exit
from scipy import stats
import sklearn
import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from operator import and_
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from datetime import datetime
#from geopy.geocoders import Nominatim
#geolocator = Nominatim()
#from matplotlib.font_manager import FontProperties
#import cartopy.crs as ccrs
#import cartopy.io.shapereader as shpreader

###############################################
# Functions
###############################################
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
###############################################

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata = '/Users/lilllianpetersen/science_fair_2018/data/'
wdvars = '/Users/lilllianpetersen/science_fair_2018/saved_vars/'
wdfigs = '/Users/lilllianpetersen/science_fair_2018/figures/'

currentMonth = datetime.now().strftime("%Y-%m-%d").split('-')[1]
#fwrite = open(wddata+'max_ndviMonths.csv','w')
#fpred = open(wddata+'crop_predictions_2018_with_actual_'+currentMonth+'.csv','w')
fweb = open(wddata+'website_with_error.csv','w')
fcorr = open(wddata+'corrs_every_index_crop_2018'+currentMonth+'.csv','w')

makePlots = False
makeFinalPlots = False
MakePredictions = True
nyears = 5
#nyears = 4

monthName=['January','Febuary','March','April','May','June','July','August','September','October','November','December']

slopesAll = np.zeros(shape = (48,6,6))
bIntAll = np.zeros(shape = (48,6,6))
maxMonthAll = np.zeros(shape = (48))
badCorrCountries = []

countryList = []
indexAll = []
cropAll = []
CorrsAll = np.zeros(shape=(48,6,6))
predictionsAll = np.zeros(shape=(48,6,6))
bestPred = np.zeros(shape=(48,6))
errorAll = np.zeros(shape=(48,6,6))
CorrAll = np.zeros(shape=(48,6,6))
predFromStdDevAll = np.zeros(shape=(48,6,6))
predInPercent = np.zeros(shape=(48,6,6))
bestPredAnom = np.zeros(shape=(48,6))
countryNamesOriginal = np.zeros(shape=(48),dtype=object)
#xMulti = np.zeros(shape = (620,6))
#ydataMulti = np.zeros(shape = (620))
xMulti = np.zeros(shape=(155,6))
ydataMulti = np.zeros(shape=(155))
iMultiCounter = -1
countryCounter = -1
harvestMonthAll = np.zeros(shape = (48),dtype = int)
harvestMonthAllName = []

for icountry in range(47):
	countryNum = str(icountry+1)
	#countryNum = '26'
	#if countryNum=='1' or countryNum=='26' or countryNum=='29' or countryNum=='22' or countryNum=='47': # South Sudan or Gabon
	if countryNum=='44' or countryNum=='13':
		continue
	if icountry!=37:
		continue
	
	SeasonOverYear = False

	cropDataDir = wddata+'africa_crop/'
	
	f = open(wddata+'africa_latlons.csv')
	for line in f:
		tmp = line.split(',')
		if tmp[0]==countryNum:
			country = tmp[1]
			countryl = country.lower()
			countryl = countryl.replace(' ','_')
			countryl = countryl.replace('-','_')
			corrMonth = tmp[5].title()
			corrMonthName = corrMonth
			corrYear = tmp[6]
			harvestMonth = float(tmp[7])
			seasons = tmp[8][:-1].split('/')
			twoSeasons = seasons[0]
			if twoSeasons != 'no':
				corrMonth1 = int(seasons[1])
				corrMonth2 = int(seasons[2])
				corrMonth = corrMonth1
			break

	########### find countries with the right growing season ###########
	Good = False
	makeFinalPlots = False
	MakePredictions = True
	#fcountries = open(wddata+'prediction_countries','r')
	#i = -1
	#for line in fcountries:
	#	i+= 1
	#	tmp=line.split(',')
	#	exit()
	#	if tmp[0][:-1]==country:
	#		Good=True
			
	fseason=open(wddata+'max_ndviMonths_final.csv','r')
	for line in fseason:
		tmp=line.split(',')
		if tmp[0]==str(countryNum):
			corrMonthtmp=tmp[2][:-1]
			if len(corrMonthtmp)>2:
				months=corrMonthtmp.split('/')
				month1=corrMonthtmp[0]
				month2=corrMonthtmp[1]
				corrMonthtmp=month1
			corrMonthtmp=int(corrMonthtmp)

			if (corrMonthtmp>=2 and corrMonthtmp<6): # July predictions
			#if (corrMonthtmp>5): # Feb predictions
				print '\nRunning',country, ' month = '+monthName[corrMonthtmp-1]
				Good=True
				break
			else:
				print country, 'has other season'
				break

	if Good==False:
		continue
		#makeFinalPlots=False
		#MakePredictions=False
	if country=='Burkina Faso': continue
		
	####################################################################
	########### load variables ###########
	ndviAnom=np.load(wdvars+country+'/ndviAnom.npy')
	eviAnom=np.load(wdvars+country+'/eviAnom.npy')
	ndwiAnom=np.load(wdvars+country+'/ndwiAnom.npy')
	
	ndviAvg=np.load(wdvars+country+'/ndviAvg.npy')
	eviAvg=np.load(wdvars+country+'/eviAvg.npy')
	ndwiAvg=np.load(wdvars+country+'/ndwiAvg.npy')
	if len(ndviAvg.shape)==3:
		ndviAvg=ndviAvg[0]	
		eviAvg=eviAvg[0]	
		ndwiAvg=ndwiAvg[0]	
		ndviAnom=ndviAnom[0]	
		eviAnom=eviAnom[0]	
		ndwiAnom=ndwiAnom[0]	

	ndviAnom2020=np.load(wdvars+country+'/2020_july/ndviAnom.npy')
	eviAnom2020=np.load(wdvars+country+'/2020_july/eviAnom.npy')
	ndwiAnom2020=np.load(wdvars+country+'/2020_july/ndwiAnom.npy')
	
	ndviAvg2020=np.load(wdvars+country+'/2020_july/ndviAvg.npy')
	eviAvg2020=np.load(wdvars+country+'/2020_july/eviAvg.npy')
	ndwiAvg2020=np.load(wdvars+country+'/2020_july/ndwiAvg.npy')
	######################################
	exit()

	print '\n',country,countryNum

	files = [filename for filename in os.listdir(cropDataDir) if filename.startswith(countryl+'-')]
	crop=[]
	for n in files:
		tmp=n.split('-')
		croptmp=tmp[1].title()
		crop.append(tmp[1].title())
	
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
		elif crop[cp]=='Green_Coffee' or crop[cp]=='Green_Coffe':
			crop[cp]='Coffee'
	
	if np.amax(cropYield)==np.amin(cropYield):
		continue
	print crop
	
	fCrops2018=open(wddata+'2018africaCrops.csv','r')
	crop2018=np.zeros(shape=(len(crop)))
	for line in fCrops2018:
		tmp=line.split(',')
		countrytmp=tmp[0]
		if countrytmp!=country:
			continue
		croptmp=tmp[1].title()
		if tmp[2][:-1]=='NA':
			crop2018[np.where(np.array(crop)==croptmp)[0]]=cropYield[0,-1]
		else:
			crop2018[np.where(np.array(crop)==croptmp)[0]]=int(tmp[2])
		
	
	### Find Crop Yield Anomaly ###
	cropYieldAnom=np.zeros(shape=(cropYield.shape))
	crop2018Anom=np.zeros(shape=(crop2018.shape))
	meanYield=np.zeros(shape=(len(crop)))
	for cp in range(len(crop)):
		meanYield[cp]=np.mean(cropYield[cp,:])
		crop2018Anom[cp]=crop2018[cp]-meanYield[cp]
		for y in range(nyears):
			cropYieldAnom[cp,y]=cropYield[cp,y]-meanYield[cp]
	
	if np.ma.amax(cropYieldAnom)==0. and np.amin(cropYieldAnom)==0:
		cropYieldAnom[:]=1
	
	countryList.append(country)
	countryCounter+=1

	if nyears==4:
		print 'what??'
		exit()
		ndviAvg=ndviAvg[:-1]
		eviAvg=eviAvg[:-1]
		ndwiAvg=ndwiAvg[:-1]
		ndviAnom=ndviAnom[:-1]
		eviAnom=eviAnom[:-1]
		ndwiAnom=ndwiAnom[:-1]
	
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

	ndwiAvg=ndwiAvgNormal

	#ndwiAnom=abs(ndwiAnom)

	#ndwiAvg=np.zeros(shape=(ndwiAvgNormal.shape))
	#mean=np.mean(ndwiAvgNormal)
	#for y in range(nyears):
	#	for m in range(12):
	#		ndwiAvg[y,m]=(1./(ndwiAvgNormal[y,m]/mean))*mean 
	
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
			plt.title(country+' NDVI Monthly Average')
			plt.grid(True)
			plt.savefig(wdfigs+country+'/ndviAvg_over_time.pdf',dpi=700)
			plt.clf()
	
	########################
	Corr=np.zeros(shape=(6,6))
	slope=np.zeros(shape=(6,6))
	bInt=np.zeros(shape=(6,6))
	data2017=np.zeros(shape=(6,6))
	
	
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
			ydataNDVI2020=np.ma.amax(ndviAvg2020)
			maxMonth[y]=np.ma.where(ndviAvg[y,:]==np.ma.amax(ndviAvg[y,:]))[0][0]
			maxMonth2020=np.ma.where(ndviAvg2020==ydataNDVI2020)
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
			Corr[0,cp]=stats.pearsonr(cropYield[cp,1:],ydataNDVI[:-1])[0]
			#x=np.arange(2013.96,2017.96)
			x=np.arange(2013.+(harvestMonth/12)+1,2013.+(harvestMonth/12)+nyears)
			ydata=cropYield[cp,1:]
			slope[0,cp],bInt[0,cp]=np.polyfit(ydataNDVI[:-1],cropYield[cp,1:],1)
			yfit=slope[0,cp]*ydataNDVI[:-1]+bInt[0,cp]
		elif corrYear=='same':
			Corr[0,cp]=stats.pearsonr(cropYield[cp,:],ydataNDVI[:])[0]
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
	#	Corr=stats.pearsonr(cropYield[cp,:],ndviAvg[0,:,7])[0]
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
		
	if corrYear=='after':
		ydataAll=np.zeros(shape=(len(crop),4)) # cp,y
	else:
		ydataAll=np.zeros(shape=(len(crop),5))
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
		Corr[0,cp]=stats.pearsonr(cropYield[cp,:],ydataNDVI)[0]
	
		if corrYear=='after':
			ydata=cropYield[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[0,cp]=stats.pearsonr(cropYield[cp,1:],ydataNDVI[:-1])[0]
			slope[0,cp],bInt[0,cp]=np.polyfit(ydataNDVI[:-1],cropYield[cp,1:],1)
			yfit=slope[0,cp]*ydataNDVI[:-1]+bInt[0,cp]
		if corrMonth!='Max':
			Corr[0,cp]=stats.pearsonr(cropYield[cp,:],ndviAvg[:,int(corrMonth)])[0]
			ydataNDVI=ndviAvg[:,int(corrMonth)]
			corrMonthName=monthName[int(corrMonth)]
			slope[0,cp],bInt[0,cp]=np.polyfit(ydataNDVI[:],cropYield[cp,:],1)
			yfit=slope[0,cp]*ydataNDVI[:]+bInt[0,cp]

		data2017[0,cp]=ydataNDVI[-1]

		ydataAll[cp,:]=ydata
	
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
			ydataEVI2020=eviAvg2020[maxMonth2020]
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
		Corr[1,cp]=stats.pearsonr(cropYield[cp,:],ydataEVI)[0]
		slope[1,cp],bInt[1,cp]=np.polyfit(ydataEVI[:],cropYield[cp,:],1)
		yfit=slope[1,cp]*ydataEVI[:]+bInt[1,cp]
	
		if corrYear=='after':
			ydata=cropYield[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[1,cp]=stats.pearsonr(cropYield[cp,1:],ydataEVI[:-1])[0]
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
			#	Corr=stats.pearsonr(cropYield[cp,1:],eviAvg[0,:-1,7])[0]
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
			ydataNDWI2020=ndwiAvg2020[maxMonth2020]
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
		Corr[2,cp]=stats.pearsonr(cropYield[cp,:],ydataNDWI)[0]
		slope[2,cp],bInt[2,cp]=np.polyfit(ydataNDWI[:],cropYield[cp,:],1)
		yfit=slope[2,cp]*ydataNDWI[:]+bInt[2,cp]
	
		if corrYear=='after':
			ydata=cropYield[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[2,cp]=stats.pearsonr(cropYield[cp,1:],ydataNDWI[:-1])[0]
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
				ydataNDVIAnom2020=ndviAnom2020[maxMonth2020]
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

		Corr[3,cp]=stats.pearsonr(cropYieldAnom[cp,:],ydataNDVIAnom)[0]
		slope[3,cp],bInt[3,cp]=np.polyfit(ydataNDVIAnom[:],cropYield[cp,:],1)
		yfit=slope[3,cp]*ydataNDVIAnom[:]+bInt[3,cp]
	
		if corrYear=='after':
			ydata=cropYieldAnom[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[3,cp]=stats.pearsonr(cropYieldAnom[cp,1:],ydataNDVIAnom[:-1])[0]
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
			ydataEVIAnom2020=eviAnom2020[maxMonth2020]
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
		Corr[4,cp]=stats.pearsonr(cropYieldAnom[cp,:],ydataEVIAnom)[0]
		slope[4,cp],bInt[4,cp]=np.polyfit(ydataEVIAnom[:],cropYield[cp,:],1)
		yfit=slope[4,cp]*ydataEVIAnom[:]+bInt[4,cp]
	
		if corrYear=='after':
			ydata=cropYieldAnom[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[4,cp]=stats.pearsonr(cropYieldAnom[cp,1:],ydataEVIAnom[:-1])[0]
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
			#	Corr=stats.pearsonr(cropYieldAnom[cp,1:],eviAnom[0,:-1,7])[0]
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
			ydataNDWIAnom2020=ndwiAnom2020[maxMonth2020]
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
		Corr[5,cp]=stats.pearsonr(cropYieldAnom[cp,:],ydataNDWIAnom)[0]
		slope[5,cp],bInt[5,cp]=np.polyfit(ydataNDWIAnom[:],cropYield[cp,:],1)
		yfit=slope[5,cp]*ydataNDWIAnom[:]+bInt[5,cp]
		#corrMonthName=corrMonth
	
		if corrYear=='after':
			ydata=cropYieldAnom[cp,1:]
			x=np.arange(2013+.14,2017+.14)
			Corr[5,cp]=stats.pearsonr(cropYieldAnom[cp,1:],ydataNDWIAnom[:-1])[0]
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
	slopesAll[countryNum]=slope[:,:]
	bIntAll[countryNum]=bInt[:,:]

	data=Counter(maxMonth)
	common=data.most_common(1)[0][0]
	maxMonthAll[countryNum]=common
	harvestMonthAll[countryCounter]=maxMonthAll[countryNum]+3
	seasonHight=int(maxMonthAll[countryNum])

	if harvestMonthAll[countryCounter]>11:
		harvestMonthAll[countryCounter]=harvestMonthAll[countryCounter]-12
	harvestMonthAll[0]=11

	print round(maxCorr,2)
	print np.ma.compress_cols(np.ma.masked_array(np.round(Corr,2),Corr==0))
	print crop[whereMaxCorrY], variables[whereMaxCorrX]
	print 'month =', round(maxMonthAll[countryNum],0)

	cropA=[]
	for i in range(len(crop)):
		cropA.append(np.array(crop)[i])
	for j in range(i+1,6):
		cropA.append('')
	cropA=np.array(cropA)
	cropAll.append(cropA)

	CorrsAll[countryCounter]=Corr
	indexAll.append(variables[whereMaxCorrX])
	
	if maxCorr<.75:
		print 'low corr'

	#if twoSeasons=='no':
	#	fwrite.write(str(countryNum)+','+country+','+str(seasonHight)+'\n')
	#else:
	#	fwrite.write(str(countryNum)+','+country+','+str(corrMonth1)+'/'+str(int(corrMonth2))+'\n')

	CorrAll[countryCounter,:,:]=Corr
	if Good==False:
		for cp in range(len(crop)):
			iMultiCounter+=1
			xMulti[iMultiCounter,0]
			for index in range(6):
				harvestMonthAllName.append(monthName[harvestMonthAll[countryCounter]])

				#fcorr.write(country+' & '+crop[cp]+' & '+variables[index]+' & '+
				#	str(Corr[index,cp])+' & '+harvestMonthAllName[countryCounter]+' \\\\ \n')

	if MakePredictions==False:
		print 'MakePredictions = False'
		continue

	if MakePredictions:
		#################################### Make Predictions ####################################
		#ndviAnom2020=np.load(wdvars+country+'/2020/ndviAnom.npy')
		#eviAnom2020=np.load(wdvars+country+'/2020/eviAnom.npy')
		#ndwiAnom2020=np.load(wdvars+country+'/2020/ndwiAnom.npy')
		#
		#ndviAvg2020=np.load(wdvars+country+'/2020/ndviAvg.npy')
		#eviAvg2020=np.load(wdvars+country+'/2020/eviAvg.npy')
		#ndwiAvg2020=np.load(wdvars+country+'/2020/ndwiAvg.npy')

		#if corrYear=='after':
		#	ndviAvg2020=ndviAvg[-1]
		#	eviAvg2020=eviAvg[-1]
		#	ndwiAvg2020Normal=ndwiAvg[-1]
		#	ndwiAvg2020=ndwiAvg2020Normal
	
		#	ndviAnom2020=ndviAnom[-1]
		#	eviAnom2020=eviAnom[-1]
		#	ndwiAnom2020=ndwiAnom[-1]

		if np.isnan(ndviAvg2020[-1])==True:
			print 'why are there nans???'
			exit()
			ndviAvg2020[-1]=ndviAvg2020[-2]
			eviAvg2020[-1]=eviAvg2020[-2]
			ndwiAvg2020[-1]=ndwiAvg2020[-2]

			ndviAnom2020[-1]=ndviAnom2020[-2]
			eviAnom2020[-1]=eviAnom2020[-2]
			ndwiAnom2020[-1]=ndwiAnom2020[-2]

		ydataForPred=np.zeros(shape=(6))
		ydataForPred[0]=np.ma.amax(ndviAvg2020)
		month2020=np.where(ndviAvg2020==ydataForPred[0])[0][0]
		ydataForPred[1]=eviAvg2020[month2020]
		ydataForPred[2]=ndwiAvg2020[month2020]

		ydataForPred[3]=ndviAnom2020[month2020]
		ydataForPred[4]=eviAnom2020[month2020]
		ydataForPred[5]=ndwiAnom2020[month2020]

		#else:
		#	ydataForPred[0]=ydataNDVI[-1]
		#	ydataForPred[1]=ydataEVI[-1]
		#	ydataForPred[2]=ydataNDWI[-1]

		#	ydataForPred[3]=ydataNDVIAnom[-1]
		#	ydataForPred[4]=ydataEVIAnom[-1]
		#	ydataForPred[5]=ydataNDWIAnom[-1]
		#	if np.isnan(ydataForPred[5])==True:
		#		print 'ydataForPred[5]=nan'
		#		ydataForPred[5]=np.ma.mean([ndwiAnom2020[maxMonth[-1]-1],ndwiAnom2020[maxMonth[-1]+2]])

		if twoSeasons!='no':
			print 'Make sure this still works: 2 seasons'
			#continue
			ydataForPred[0]=np.amax(ndviAvg2020)
			month2020=np.where(ndviAvg2020==ydataForPred[0])[0][0]

			ydataForPred[1]=eviAvg2020[month2020]
			ydataForPred[2]=ndwiAvg2020[month2020]

			ydataForPred[3]=ndviAnom2020[month2020]
			ydataForPred[4]=eviAnom2020[month2020]
			ydataForPred[5]=ndwiAnom2020[month2020]
			


		#############################################################################
		# data for plot 
		#############################################################################
		predFromStdDev=np.zeros(shape=(len(crop),6))
		#prodFromStdDev=np.zeros(shape=(6,len(crop)))
		bar_width=0.2
		for cp in range(len(crop)):
			for index in range(6):

				#### Predict 2020 yield ###
				yieldPred=slope[index,cp]*ydataForPred[index]+bInt[index,cp]
				prod=crop2018[cp]
				predAnom=yieldPred-meanYield[cp]
				prodAnom=prod-meanYield[cp]
		
				stdDevYield=np.std(cropYield[cp])
				predFromStdDev[cp,index]=predAnom/stdDevYield
				#prodFromStdDev[index,cp]=prodAnom/stdDevYield
				###########################

				if variables[index]=='ndviAvg':
					xdataMonthly=np.ma.compressed(ndviAvg)
					xdata=np.ma.compressed(ydataNDVI)
					xdata2020=ydataNDVI2020
				elif variables[index]=='eviAvg':
					xdataMonthly=np.ma.compressed(eviAvg)
					xdata=np.ma.compressed(ydataEVI)
					xdata2020=ydataEVI2020
				elif variables[index]=='ndwiAvg':
					xdataMonthly=np.ma.compressed(ndwiAvg)
					xdata=np.ma.compressed(ydataNDWI)
					xdata2020=ydataNDWI2020
		
				elif variables[index]=='ndviAnom':
					xdata=np.ma.compressed(ydataNDVIAnom)
					xdata2020=ydataNDVIAnom2020
				elif variables[index]=='eviAnom':
					xdata=np.ma.compressed(ydataEVIAnom)
					xdata2020=ydataEVIAnom2020
				elif variables[index]=='ndwiAnom':
					xdata=np.ma.compressed(ydataNDWIAnom)
					xdata2020=ydataNDWIAnom2020
		
				if corrYear=='after':
					x=np.arange(2013.+(harvestMonth/12),2017.+(harvestMonth/12))
					ydata=cropYield[cp,1:]
					if variables[index][-1]=='m': # if best predictor is an anomaly
						ydata=cropYieldAnom[cp,1:]
				else:
					#print 'corrYear = Same:', 'Load 2018 Data' 
					ydata=cropYield[cp]
					x=np.arange(2013.+(harvestMonth/12),2018.+(harvestMonth/12))
					if variables[index][-1]=='m': # if best predictor is an anomaly
						ydata=cropYieldAnom[cp]
		
				###########################################
				# Prediction Plots
				###########################################
		
				if makeFinalPlots:

					if not os.path.exists(wdfigs+'current_harvest/'+country):
						os.makedirs(wdfigs+'current_harvest/'+country)

					if variables[index]=='ndviAvg':
				
						if corrYear=='same':
							satellitePlot=np.zeros(shape=(xdataMonthly.shape[0]+ndviAvg2020.shape[0]))
							satellitePlot[:xdataMonthly.shape[0]] = xdataMonthly # data 2013-2018
							satellitePlot[xdataMonthly.shape[0]:] = ndviAvg2020 # data 2020
							#satellitePlot[xdataMonthly.shape[0]:]=ndviAvg2018
							satellitePlot2019 = ndviAvg2020
						else:
							satellitePlot=xdataMonthly
							satellitePlot2019 = ndviAvg2020[:12] # 2019 data
				
						xTimeSmall=np.zeros(shape=(ndviAvg2020.shape))
						for m in range(len(xTimeSmall)):
							xTimeSmall[m]=2020+(m+.5)/12
				
						if corrYear=='same':
							xtimeNew=np.zeros(shape=(xtime.shape[0]+ndviAvg2020.shape[0]))
							xtimeNew[:xtime.shape[0]]=xtime
							xtimeNew[xtime.shape[0]:]=xTimeSmall
							xtime2019 = xTimeSmall
						else:
							xtimeNew = xtime
							xtime2019 = np.zeros(shape=(12))
							for m in range(12):
								xtime2019[m]=(2019) + (m+0.5)/12

							xtime2019 = np.ma.compressed(np.ma.masked_array(xtime2019, satellitePlot2019==0))
							satellitePlot2019 = np.ma.compressed(np.ma.masked_array(satellitePlot2019, satellitePlot2019==0))

						props = dict(boxstyle='round', facecolor='white', alpha=1)

						#error=abs(crop2018[cp]-yieldPred)/crop2018[cp]*100
				
						plt.clf()
						fig, ax1 = plt.subplots()
						ax2 = ax1.twinx()
						ax3 = ax1.twinx()
						ax1.grid(True)
					
						ax2.bar(x,ydata,bar_width,color='g',label=crop[cp]+' Production')
						ax2.bar(x[-1]+1,crop2018[cp],bar_width,color='g')
						ax2.bar(x[-1]+3,yieldPred,bar_width,color='m',label='Predicted Production')

						ax2.legend(loc='upper right')
						ax2.tick_params(axis='y',colors='g')
						ax2.set_ylabel(crop[cp]+' Production, Gigatonnes',color='g')
					
						ax2.set_ylim([np.ma.amin(ydata)*.96,np.amax([np.amax(ydata),np.amax(yieldPred)])*1.02])
						ax1.set_ylim([np.ma.amin(satellitePlot*100)*.9,np.amax( [np.amax(satellitePlot*100),np.amax(satellitePlot2019*100)] )*1.15])
					
						ax1.plot(xtimeNew,satellitePlot*100,'b*-',label='Monthly NDVI')
						ax1.plot(xtime2019,satellitePlot2019*100,'b*-')
						if country=='Ethiopia' or country=='Egypt':
							satelliteSince2018 = np.load(wdvars+'/'+country+'/since2018/ndviAvg.npy')
							ax1.plot(xtimeNew[xtimeNew>2017]+1, satelliteSince2018[0]*100, 'b*-')
							ax1.plot(xtimeNew[xtimeNew>2017]+2, satelliteSince2018[1]*100, 'b*-')

						ax1.legend(loc='upper left')
				
						ax1.set_ylabel('NDVI Monthly Average *100',color='b')
						ax1.tick_params(axis='y',colors='b')

						ax3.text(2015.5, .93, 'Corr='+str(round(Corr[index,cp],2)),bbox=props)
						ax3.axis('off')
					
						plt.title(country+': '+variablesTitle[index][:4]+' and '+crop[cp]+' Prod')
						#plt.savefig(wdfigs+country+'/pred_monthly_'+variables[index]+'_avg_with_'+crop[cp]+'.pdf',dpi=700)
						plt.savefig(wdfigs+'current_harvest/'+country+'/'+country+'_'+variables[index]+'_monthly_with_'+crop[cp]+'.pdf',dpi=700)
					
				###########################################################
				
					if variables[index]=='ndviAvg' or variables[index]=='eviAvg' or variables[index]=='ndwiAvg': # Bar Prediction Plot
						bar_width = 0.27
				
						if corrYear=='same':
							print 'Add 2018 data to satellite'
							satellitePlot=np.zeros(shape=(xdata.shape[0]+1))
							satellitePlot[:xdata.shape[0]]=xdata
							satellitePlot[xdata.shape[0]:]=ydataForPred[index]
							satellitePlot2020 = xdata2020
						else:
							satellitePlot = xdata
							satellitePlot2020 = xdata2020
				
						if variables[index]=='ndwiAvg':
							satellitePlot=-1*satellitePlot
				
						if corrYear=='same':
							xNDVI=np.arange(2013-.14,2013+nyears+1-.14)
							x=np.arange(2013+.14,2013+nyears+.14)
							xNDVI2020 = 2020-0.14
						else:
							xNDVI=np.arange(2013-.14,2013+nyears-.14)
							x=np.arange(2013+.14,2013+nyears+.14)
							xNDVI2020 = 2019-0.14
				
						plt.clf()
						fig, ax2 = plt.subplots()
						ax1 = ax2.twinx()
						ax3 = ax2.twinx()
				
						if country=='Ethiopia':
							ax1.set_ylim([np.ma.amin(ydata)*.96,np.amax([np.ma.amax(ydata),yieldPred])*1.035]) # for Ethiopia
							ax2.set_ylim([np.ma.amin(satellitePlot*100)*.65,np.amax([np.ma.amax(satellitePlot*100),satellitePlot2020*100])*1.15])
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
							ax1.set_ylim([np.ma.amin(ydata)*.72,np.ma.amax(ydata)*1.15]) 
							ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.95,np.ma.amax(satellitePlot*100)*1.05])
							ax2.set_yticks([47,50,53,56,59,62])
						else:
							ax1.set_ylim([np.ma.amin([np.ma.amin(ydata),yieldPred])*.9,np.ma.amax([np.ma.amax(ydata),yieldPred])*1.1]) 
							ax2.set_ylim([np.ma.amin(satellitePlot*100)*0.9,np.ma.amax(satellitePlot*100)*1.1])
				
					
						if variablesTitle[index]=='NDWI Avg':
							ax2.set_ylabel('Max '+variablesTitle[index][:4]+' Month Average *100, flipped',color='b')
						else:
							ax2.set_ylabel('Max '+variablesTitle[index][:4]+' Month Average *100',color='b')
				
						label=crop[cp]+' Production'
						if corrYear=='after':
							ax1.bar(x[:-1],ydata,bar_width,color='g',label=label)
							ax1.bar(2019.14,yieldPred,bar_width,color='m',label='Predicted Production')
							label='Predicted production'
							#ax1.bar(x[-1],yieldPred,bar_width,color='m',label=label)
							ax1.bar(2017.14,crop2018[cp],bar_width,color='g')
						else:
							ax1.bar(x,ydata,bar_width,color='g',label=label)
							label='Predicted production'
							#ax1.bar(x[-1]+1,yieldPred,bar_width,color='m',label=label)
							ax1.bar(x[-1]+1+.14,crop2018[cp],bar_width,color='g')
							ax1.bar(2020.14,yieldPred,bar_width,color='m',label='Predicted Production')
						ax1.tick_params(axis='y',colors='g')
						ax2.tick_params(axis='y',colors='b')
						ax2.bar(xNDVI,(satellitePlot*100),bar_width,color='b',label='Max '+variablesTitle[index])
						ax2.bar(xNDVI2020,satellitePlot2020*100,bar_width,color='b')
		
						#error=abs(crop2018[cp]-yieldPred)/crop2018[cp]*100
					
						#corrYear='after'
						# Corr text
						if corrYear=='same':
							props = dict(boxstyle='round', facecolor='white', alpha=1)
							ax3.text(2015.2, .93, 'Corr='+str(round(Corr[index,cp],2)),bbox=props)
						else:
							props = dict(boxstyle='round', facecolor='white', alpha=1)
							ax3.text(2014.4, .93, 'Corr='+str(round(Corr[index,cp],2)),bbox=props)
				
						ax1.set_ylabel('Production, Gigatonnes',color='g')
						ax3.axis('off')
						#plt.title(country+': '+variablesTitle[index][:4]+' and '+crop[cp]+' Prod, Pred Error = '+str(round(error,1))+' %')
						plt.title(country+': Max '+variablesTitle[index][:4]+' and '+crop[cp]+' Prod')
						ax2.grid(True)
						if corrYear=='after':
							ax1.set_xticks(range(2013,2019))
							ax2.set_xticks(range(2013,2019))
						else:
							ax1.set_xticks(range(2013,2020))
							ax2.set_xticks(range(2013,2020))
						ax1.legend(loc='upper right')
						ax2.legend(loc='upper left')
						plt.savefig(wdfigs+'current_harvest/'+country+'/'+country+'_'+variables[index]+'_with_'+crop[cp]+'.pdf',dpi=700)
						#if error>1000:
						#	print crop[cp],variables[index],error
						#	exit()
				
				###########################################################
				
					if variables[index]=='ndviAnom' or variables[index]=='eviAnom' or variables[index]=='ndwiAnom': # Bar Prediction Plot
						bar_width = 0.27
				
						if corrYear=='same':
							print 'Add 2018 data to satellite'
							satellitePlot=np.zeros(shape=(xdata.shape[0]+1))
							satellitePlot[:xdata.shape[0]]=xdata
							satellitePlot[xdata.shape[0]:]=ydataForPred[index]
							satellitePlot2020 = xdata2020
						else:
							satellitePlot = xdata
							satellitePlot2020 = xdata2020
				
						if variables[index]=='ndwiAvg':
							satellitePlot=-1*satellitePlot
				
						if corrYear=='same':
							xNDVI=np.arange(2013-.14,2013+nyears+1-.14)
							x=np.arange(2013+.14,2013+nyears+.14)
							xNDVI2020 = 2020-0.14
						else:
							xNDVI=np.arange(2013-.14,2013+nyears-.14)
							x=np.arange(2013+.14,2013+nyears+.14)
							xNDVI2020 = 2019-0.14
				
						plt.clf()
						fig, ax2 = plt.subplots()
						ax1 = ax2.twinx()
						ax3 = ax2.twinx()
				
						ydataAbs1=np.ma.amax(abs(ydata))
						ydataAbs2=abs(predAnom)
						ydataAbs=np.ma.amax([ydataAbs1,ydataAbs2])
				
						ax1.set_ylim([ydataAbs*-1.2,ydataAbs*1.5]) 
						ax2.set_ylim([np.ma.amax(abs(satellitePlot*100))*-1.2,np.ma.amax(abs(satellitePlot*100))*1.5])
				
						ax2.set_ylabel(variablesTitle[index][:4]+' Anomaly *100',color='b')
		
						#error=abs(crop2018[cp]-yieldPred)/crop2018[cp]*100
						#if error>1000:
						#	exit()
				
						label=crop[cp]+' Production'
						if corrYear=='after':
							ax1.bar(x[:-1],ydata+1000,bar_width,bottom=-1000,color='g',label=label)
							ax1.bar(2017.14,crop2018Anom[cp]+1000,bar_width,bottom=-1000,color='g')
							ax1.bar(2019.14,predAnom+1000,bar_width,bottom=-1000,color='g')
						else:
							ax1.bar(x,ydata+1000,bar_width,bottom=-1000,color='g',label=label)
							label='Predicted Production'
							ax1.bar(x[-1]+1+.14,crop2018Anom[cp]+1000,bar_width,bottom=-1000,color='g')
							ax1.bar(2020.14,predAnom+1000,bar_width,bottom=-1000,color='g')
						ax1.tick_params(axis='y',colors='g')
						ax2.tick_params(axis='y',colors='b')
						ax2.bar(xNDVI,satellitePlot*100+100,bar_width,bottom=-100,color='b',label=variablesTitle[index])
						ax2.bar(xNDVI2020,satellitePlot2020*100+100,bar_width,bottom=-100,color='b')
					
						# Corr text
						props = dict(boxstyle='round', facecolor='white', alpha=1)
						if corrYear=='same':
							props = dict(boxstyle='round', facecolor='white', alpha=1)
							if country=='Rwanda':
								ax3.text(2014.8, .93, 'Corr='+str(round(Corr[index,cp],3)),bbox=props)
							else:
								ax3.text(2014.8, .93, 'Corr='+str(round(Corr[index,cp],2)),bbox=props)
						else:
							props = dict(boxstyle='round', facecolor='white', alpha=1)
							if country=='Rwanda':
								ax3.text(2014.4, .93, 'Corr='+str(round(Corr[index,cp],3)),bbox=props)
							else:
								ax3.text(2014.4, .93, 'Corr='+str(round(Corr[index,cp],2)),bbox=props)
				
						ax3.axis('off')
						ax1.set_ylabel('Production Anomaly, Gigatonnes',color='g')
						#plt.title(country+': '+variablesTitle[index][:4]+' and '+crop[cp]+' Prod, Pred Error = '+str(round(error,1))+' %')
						plt.title(country+': '+variablesTitle[index][:4]+' and '+crop[cp]+' Prod, Pred = '+str(round(predFromStdDev[cp,index],1))+' std Dev')
						ax2.grid(True)
						if corrYear=='same':
							ax1.set_xticks(range(2013,2020))
						else:
							ax1.set_xticks(range(2013,2019))
						ax1.legend(loc='upper right')
						ax2.legend(loc='upper left')
						#plt.savefig(wdfigs+country+'/'+country+'_pred_monthly_'+variables[index]+'_avg_with_'+crop[cp]+'.pdf',dpi=700)
						plt.savefig(wdfigs+'current_harvest'+'/'+country+'/'+country+'_'+variables[index]+'_with_'+crop[cp]+'.pdf',dpi=700)

					#print str(np.round(error,2))+'%',variables[index],crop[cp]
					#errorAll[countryCounter,index,cp]=error

					#if Corr[index,cp]>.75:
					#	print 'good corr'
					#else:
					#	badCorrCountries.append(country)
		
				predictionsAll[countryCounter,index,cp]=yieldPred
				predFromStdDevAll[countryCounter,index,cp]=predFromStdDev[cp,index]

				###########################################
				# Write to file for interactive map
				###########################################
				
				harvestMonthAllName.append(monthName[harvestMonthAll[countryCounter]])
			
			bestCorr = np.amax(Corr,axis=0)[cp]
			whereBestCorr = np.where(Corr == bestCorr)[0][0]
			bestPred[countryCounter,cp] = predictionsAll[countryCounter,whereBestCorr,cp]
			bestPredAnom[countryCounter,cp] = predFromStdDevAll[countryCounter,whereBestCorr,cp]
			countryNamesOriginal[countryCounter] = country
	
	if country=='Mozambique': exit()
	
	ncp = len(crop)
	predInPercent[countryCounter,:,:ncp] = (predictionsAll[countryCounter,:,:ncp] - np.mean(cropYield,axis=1)) / np.mean(cropYield,axis=1) *100

countryAnomOriginal = np.zeros(shape=48)
for icountry in range(48):
	countryAnomOriginal[icountry] = np.mean(predFromStdDevAll[icountry,CorrAll[icountry]>0.5])

predPercentAvg = np.zeros(shape=48)
for icountry in range(48):
	predPercentAvg[icountry] = np.mean(predInPercent[icountry,CorrAll[icountry]>0.5])

#countryAnom = np.array(np.ma.mean(np.ma.masked_array(bestPredAnom[:,:5],bestPredAnom[:,:5]==0),axis=1))
#countryNames = countryNames[countryAnom!=0]
#countryAnom = countryAnom[countryAnom!=0]
#predFromStdDevAll = np.ma.masked_array(predFromStdDevAll, predFromStdDevAll==0)
#countryAnom = np.ma.mean(np.ma.mean(predFromStdDevAll,axis=1),axis=1)
#
StdDev = False
Percents = True
if StdDev:
	sort = countryAnomOriginal.argsort()
	countryAnom = np.array(countryAnomOriginal)[sort]
	countryNames = np.array(countryNamesOriginal)[sort]
	countryNames[countryNames=='Central Africa Republic'] = 'Cen. African Rep.'
	countryNames = countryNames[np.isnan(countryAnom)==False]
	countryAnom = countryAnom[np.isnan(countryAnom)==False]

	color = np.zeros(shape=len(countryNames),dtype=object)
	color[:] = 'blue'
	color[countryAnom<-0.5] = 'red'
	color[countryAnom>0.5] = 'green'

if Percents:
	sort = predPercentAvg.argsort()
	predPercentAvg= np.array(predPercentAvg)[sort]
	countryNames = np.array(countryNamesOriginal)[sort]
	countryNames[countryNames=='Central Africa Republic'] = 'Cen. African Rep.'
	countryNames = countryNames[np.isnan(predPercentAvg)==False]
	predPercentAvg = predPercentAvg[np.isnan(predPercentAvg)==False]

	color = np.zeros(shape=len(countryNames),dtype=object)
	color[:] = 'blue'
	color[predPercentAvg<-20] = 'red'
	color[predPercentAvg>20] = 'green'

plt.clf()
fig, ax = plt.subplots()
if StdDev:
	ax.barh(np.arange(len(countryNames)), width=countryAnom[::-1], height=0.8, color=color[::-1],zorder=3)
if Percents:
	ax.barh(np.arange(len(countryNames)), width=predPercentAvg[::-1], height=0.8, color=color[::-1],zorder=3)
ax.set_yticks(np.arange(-0.5,len(countryNames)+0.5), minor=True)
ax.set_xlabel('Predicted Production Anomaly, % Difference from Average')
ax.grid(which='minor', zorder=0)
ax.plot([0,0],[-5,len(countryNames)+5],'k-',linewidth=3,zorder=3)
ax.set_xlim([-80,40])
ax.set_ylim([-1,len(countryNames)])
plt.gcf().subplots_adjust(left=0.21, right=0.96)
plt.title('Crop4cast Predictions, updated July 2020',fontsize=17)
plt.yticks(np.arange(0,len(countryNames)),countryNames[::-1])
#plt.savefig(wdfigs+'current_harvest/predicted_yield_percents.pdf')
plt.savefig(wdfigs+'current_harvest/crop4cast_predicted_yields_july.pdf')
plt.savefig(wdfigs+'current_harvest/crop4cast_predicted_yields_july.jpg',dpi=200)
exit()

##################
# Map
##################
colors = [(255,0,0),(255, 128, 0), (255, 255, 0), (255, 255, 255), (153, 255, 51), (0, 204, 102), (51, 51, 255)]
my_cmap = make_cmap(colors,bit=True)
shapename = 'admin_0_countries'
countries_shp = shpreader.natural_earth(resolution='110m',
	category='cultural', name=shapename)

plt.clf()
cmapArray = my_cmap(np.arange(256))
cmin = np.amin(predPercentAvg)
cmax = cmin*-1 
y1=0
y2=255

fig = plt.figure(figsize=(10, 8))
MinMaxArray=np.ones(shape=(3,2))
subPlot1 = plt.axes([0.61, 0.07, 0.2, 0.8])
MinMaxArray[0,0]=cmin
MinMaxArray[1,0]=cmax
plt.imshow(MinMaxArray,cmap=my_cmap)
plt.colorbar(label='Predicted Yield Anomaly, % Difference from Average')

ax = plt.axes([0.05,0.05,0.8,0.85],projection=ccrs.PlateCarree())
ax.set_extent([-19, 53, -37, 39], ccrs.PlateCarree())
ax.coastlines()

for country in shpreader.Reader(countries_shp).records():
	cName=country.attributes['NAME_LONG']

	ax.add_geometries(country.geometry, ccrs.PlateCarree(), edgecolor='black', facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],label=cName)

	if cName[-6:]=='Ivoire': cName="Ivory Coast"
	if cName=='Democratic Republic of the Congo': cName='DRC'
	if cName=='Republic of the Congo': cName='Congo'
	if cName=='eSwatini': cName='Swaziland'
	if cName=='The Gambia': cName='Gambia'
	if cName=='Somaliland': cName='Somalia'
	if cName=='Central African Rupublic': cName='Cen. African Rep.'
	if np.amax(cName==countryNames)==0: exit() 
	exit()

	c=np.where(cName==countrycosted)[0][0]
	x=factoryPctOne[0,c]
	y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
	icmap=min(255,int(round(y,1)))
	icmap=max(0,int(round(icmap,1)))

	if x!=0:
		size = 15*(1+x/cmax)
		plt.plot(capitalLatLon[1,c], capitalLatLon[0,c], marker='*', markersize=size, color=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]], markeredgewidth=1.5, markeredgecolor='k')
		factoryNumOne+=1
	if x==0:
		plt.plot(capitalLatLon[1,c], capitalLatLon[0,c], marker='*', markersize=7, color='darkred')


for icoast in range(9,len(countrycosted)):
	x=factoryPctOne[0,icoast]
	y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
	icmap=min(255,int(round(y,1)))
	icmap=max(0,int(round(icmap,1)))
	if x!=0:
		size = 15*(0.8+x/cmax)
		plt.plot(capitalLatLon[1,icoast], capitalLatLon[0,icoast], marker='o', markersize=size, color=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]], markeredgewidth=1.5, markeredgecolor='k')
		IntlNumOne+=1
	if x==0:
		plt.plot(capitalLatLon[1,icoast], capitalLatLon[0,icoast], marker='o', markersize=7, color='darkred')

local = str(int(np.round(100 * np.sum(factoryPctOne[0,:9]) / np.sum(factoryPctOne[0,:]),0)))
intl = str(int(np.round(100 * np.sum(factoryPctOne[0,9:]) / np.sum(factoryPctOne[0,:]),0)))

plt.title('Production of Treatment by Factory and Port', fontsize=18)
plt.legend(loc = 'lower left')
plt.text(-15,-10,str(factoryNumOne)+' Factories Open\n'+str(IntlNumOne)+' Ports Open\n'+local+'% Produced Locally', bbox=dict(fc="none", boxstyle="round"), size = 10)

plt.savefig(wdfigs+'current_factories_demand/'+Ltitles[L]+'/geographical/Export_map.pdf')


exit()
			
				#fpred.write(country+' & '+crop[cp]+' & '+variables[index]+' & '+str(round(Corr[index,cp],3))+' & '+
			#		str(int(round(predictionsAll[countryCounter,index,cp],0)))+' & '+str(int(crop2018[cp]))+' & '+str(round(errorAll[countryCounter,index,cp],2))+' \\\\ \n')

	#if country=='DR Congo':
	#	country='Democratic Republic of the Congo'
	#if country=='Central Africa Republic':
	#	country='Central African Republic'
	#if country=='Tanzania':
	#	country='United Republic of Tanzania'
	#if country=='Libya':
	#	country='Libyan Arab Jamahiriya'
	#		
	#fweb.write(country+','+crop[whereMaxCorrY]+','+str(round(maxCorr,3))+','+
	#				str(round(predFromStdDev[whereMaxCorrY,whereMaxCorrX],2))+','+str(round(errorAll[countryCounter,whereMaxCorrX,whereMaxCorrY],2))+' \n')

				#print(country+' & '+crop[cp]+' & '+variables[index]+' & '+str(round(Corr[index,cp],3))+' & '+str(int(predictionsAll[countryCounter,index,cp]))+' & '+str(int(crop2018[cp]))+' & '+str(round(errorAll[countryCounter,index,cp],2))+' & '+' \\\\')

#########################################################
# Box and Whisker Plots
#########################################################
mask=np.zeros(shape=(predictionsAll.shape))
mask[errorAll==0]=1
predictionsAllM=np.ma.masked_array(predictionsAll,mask)
errorAllM=np.ma.masked_array(errorAll,mask)
errorAllM[35,:,:][errorAllM[35,:,:]>100]=100

########## Index ##########
boxIndices=np.zeros(shape=(54,6))
for i in range(6):
	boxIndices[:,i]=np.ma.compressed(errorAllM[:,i,:])

medians=np.median(boxIndices,axis=0)
sort=np.argsort(medians)[::-1]
sort1=sort+1

indicesSorted=np.array(variablesTitle)[sort]
boxIndicesSorted=boxIndices[:,sort]

labels=list(indicesSorted)
for i in range(len(labels)):
	labels[i]=indicesSorted[i]+' ('+str(len(boxIndicesSorted[:,i]))+')'

widths=[]
for i in range(len(labels)):
	widths.append(0.5)
fig.clear()
plt.cla()
plt.close()
plt.clf()
plt.figure(25,figsize=(7,3))
ax = plt.subplot(1,1,1)
ax.set_position([.22,.15,.70,.75])
ax.set_aspect(8)
ax.set_xlim([-3,103])
ax.boxplot(boxIndices[:,sort], 0, '', vert=0, widths=widths, whis=[12.5,87.5])
plt.title('Prediction Errors by Index')
plt.yticks([1,2,3,4,5,6], labels)
plt.xlabel('Percent Error from Predicted to Actual Production')
plt.grid(axis='x')
plt.savefig(wdfigs+'box_wisker_indices.pdf')
###########################

########### Crop ###########
cropAll1=list(cropAll)
for i in range(4):
	cropAll1.append(['','','','','',''])
cropAll1=np.array(cropAll1)
uniqueCrops=np.unique(cropAll1)
crops=[]
lenwhere=np.zeros(shape=8,dtype=int)
j=-1
for i in range(len(uniqueCrops)):
	if len(np.where(cropAll1==uniqueCrops[i])[0])>=8 and uniqueCrops[i]!='':
		j+=1
		lenwhere[j]=len(np.where(cropAll1==uniqueCrops[i])[0])
		crops.append(uniqueCrops[i])

boxCrops=np.zeros(shape=(150,len(crops)))
#boxCrops=np.empty(len(crops), dtype=object)
for i in range(len(crops)):
	tmp=np.zeros(shape=150)
	for j in range(6):
		tmp[j*20:j*20+lenwhere[i]]=errorAllM[:,j,:][cropAll1==crops[i]]
	boxCrops[:,i]=tmp
boxCrops=np.ma.masked_array(boxCrops,np.round(boxCrops,5)==0)
boxCropsGood = [[y for y in row if y] for row in boxCrops.T]
boxCropsGood=np.array(boxCropsGood)

medians=np.zeros(shape=8)
for i in range(8):
	medians[i]=np.median(boxCropsGood[i])
sort=np.argsort(medians)[::-1]
sort1=sort+1

boxCropsSorted=[]
for i in range(8):
	boxCropsSorted.append(boxCropsGood[sort[i]])
cropsSorted=np.array(crops)[sort]

labels=list(cropsSorted)
for i in range(len(labels)):
	labels[i]=cropsSorted[i]+' ('+str(len(boxCropsSorted[i]))+')'

widths=[]
for i in range(len(labels)):
	widths.append(0.5)
plt.cla()
plt.close()
plt.clf()
plt.figure(25,figsize=(7,3.9))
ax = plt.subplot(1,1,1)
ax.set_position([.22,.15,.70,.75])
ax.set_aspect(8)
ax.set_xlim([-3,103])
ax.boxplot(boxCropsSorted, 0, '', vert=0, widths=widths, whis=[12.5,87.5])
plt.title('Prediction Errors by Crop')
plt.yticks([1,2,3,4,5,6,7,8], labels)
plt.xlabel('Percent Error from Predicted to Actual Production')
plt.grid(axis='x')
plt.savefig(wdfigs+'box_wisker_crop.pdf')
###########################

######### Country #########
errorAllC=np.zeros(shape=(21,6,6))
countriesM=[]
j=-1
for i in range(44):
	if np.ma.is_masked(np.amax(errorAllM[i,:,:])):
		continue
	j+=1
	errorAllC[j]=np.ma.array(errorAllM[i])
	countriesM.append(countryList[i])
errorAllC=np.ma.masked_array(errorAllC,errorAllC==0)	
errorAllC=np.reshape(errorAllC,[21,36])
boxCountry = [[y for y in row if y] for row in errorAllC]

medians=np.zeros(shape=21)
for i in range(21):
	medians[i]=np.median(boxCountry[i])
sort=np.argsort(medians)[::-1]

boxCountrySorted=[]
for i in range(21):
	boxCountrySorted.append(boxCountry[sort[i]])
countriesSorted=np.array(countriesM)[sort]

labels=list(countriesSorted)
for i in range(len(labels)):
	labels[i]=countriesSorted[i]+' ('+str(len(boxCountrySorted[i]))+')'

widths=[]
for i in range(len(labels)):
	widths.append(0.5)
plt.cla()
plt.close()
plt.clf()
plt.figure(25,figsize=(7,10))
ax = plt.subplot(1,1,1)
ax.set_position([.22,.15,.70,.75])
ax.set_aspect(8)
ax.set_xlim([-3,103])
ax.boxplot(boxCountrySorted, 0, '', vert=0, widths=widths, whis=[12.5,87.5])
plt.title('Prediction Errors by Country')
plt.yticks(range(1,22), labels)
plt.xlabel('Percent Error from Predicted to Actual Production')
plt.grid(axis='x')
plt.savefig(wdfigs+'box_wisker_countries.pdf')
###########################

######## latitudes ########

countryLats=np.zeros(shape=(len(countriesM)))
latGroups=np.zeros(shape=(len(countriesM)))
for i in range(len(countriesM)):
	f=open(wddata+'countries_by_latitude.csv','r')
	country=countriesM[i]
	for line in f:
		tmp=line.split(',')
		if tmp[3][:-1]==country:
			countryLats[i]=float(tmp[1])
			countryLat=float(tmp[1])
			continue
	
	if countryLat<-20:
		latGroups[i]=0
	elif countryLat<-10:
		latGroups[i]=1
	elif countryLat<0:
		latGroups[i]=2
	elif countryLat<10:
		latGroups[i]=3
	elif countryLat<20:
		latGroups[i]=4
	else:
		latGroups[i]=5

boxLatsA=np.zeros(shape=(100,6))
for i in range(21):
	for j in range(6):
		boxLatsA[:len(np.ma.compressed(errorAllC[latGroups==j])),j]=np.ma.compressed(errorAllC[latGroups==j])
boxLatsA=np.ma.masked_array(boxLatsA,np.round(boxLatsA,5)==0)
boxLats= [[y for y in row if y] for row in boxLatsA.T]

medians=np.zeros(shape=6)
for i in range(6):
	medians[i]=np.median(boxLats[i])
sort=np.argsort(medians)[::-1]

boxLatsSorted=[]
for i in range(6):
	boxLatsSorted.append(boxLats[sort[i]])
latsSorted=np.array(countryLats)[sort]

lats=['35S-20S','20S-10S','10S-0N','0N-10N','10N-20N','20-38N']
latsSorted=list(np.array(lats)[sort])
labels=list(lats)
labelsUnsorted=list(lats)
for i in range(len(labels)):
	labels[i]=latsSorted[i]+' ('+str(len(boxLatsSorted[i]))+')'
	labelsUnsorted[i]=lats[i]+' ('+str(len(boxLats[i]))+')'

widths=[]
for i in range(len(labels)):
	widths.append(0.5)
plt.cla()
plt.close()
plt.clf()
plt.figure(25,figsize=(7,3))
ax = plt.subplot(1,1,1)
ax.set_position([.22,.15,.70,.75])
ax.set_aspect(8)
ax.set_xlim([-3,103])
ax.boxplot(boxLats, 0, '', vert=0, widths=widths, whis=[12.5,87.5])
plt.title('Prediction Errors by Latitude')
plt.yticks(range(1,7), labelsUnsorted)
plt.xlabel('Percent Error from Predicted to Actual Production')
plt.grid(axis='x')
plt.savefig(wdfigs+'box_wisker_lats.pdf')
###########################

######## Relative High-low ########
prodFromStdDevAll=np.ma.masked_array(prodFromStdDevAll,prodFromStdDevAll==0)

countryStd=np.zeros(shape=(21,6,6))
errorAll1=np.zeros(shape=(21,6,6))
stdGroups=-9999*np.ones(shape=(21,6,6))
icountry=-1
for c in range(48):
	if np.ma.is_masked(np.amax(errorAllM[c])):
		continue
	icountry+=1
	countryStd[icountry,:,:]=prodFromStdDevAll[c,:,:]
	errorAll1[icountry]=errorAllM[c,:,:]
	for cp in range(6):
		for i in range(6):
			if np.ma.is_masked(errorAllM[c,i,cp]):
				continue
			std=countryStd[icountry,i,cp]
			
			if std<-1:
				stdGroups[icountry,i,cp]=0
			elif std<-0.5:
				stdGroups[icountry,i,cp]=1
			elif std<0:
				stdGroups[icountry,i,cp]=2
			elif std<0.5:
				stdGroups[icountry,i,cp]=3
			elif std<1:
				stdGroups[icountry,i,cp]=4
			else:
				stdGroups[icountry,i,cp]=5
errorAll1=np.ma.masked_array(errorAll1,errorAll1==0)
countryStd=np.ma.masked_array(countryStd,countryStd==0)
stdGroups=np.ma.masked_array(stdGroups,stdGroups==-9999)

boxStdA=np.zeros(shape=(150,6))
for i in range(21):
	for j in range(6):
		boxStdA[:len(np.ma.compressed(errorAll1[stdGroups==j])),j]=np.ma.compressed(errorAll1[stdGroups==j])
boxStdA=np.ma.masked_array(boxStdA,np.round(boxStdA,5)==0)
boxStd= [[y for y in row if y] for row in boxStdA.T]

medians=np.zeros(shape=6)
for i in range(6):
	medians[i]=np.median(boxStd[i])
sort=np.argsort(medians)[::-1]

boxStdSorted=[]
for i in range(6):
	boxStdSorted.append(boxStd[sort[i]])

labels1=['<-1','-1-(-0.5)','-0.5-0','0-.5','0.5-1','>1']
stdSorted=list(np.array(labels1)[sort])
labels=list(labels1)
labelsUnsorted=list(labels1)
for i in range(len(labels)):
	labels[i]=stdSorted[i]+' ('+str(len(boxStdSorted[i]))+')'
	labelsUnsorted[i]=labels1[i]+' ('+str(len(boxStd[i]))+')'

widths=[]
for i in range(len(labels)):
	widths.append(0.5)
plt.cla()
plt.close()
plt.clf()
plt.figure(25,figsize=(7,3))
ax = plt.subplot(1,1,1)
ax.set_position([.22,.15,.70,.75])
ax.set_xlim([-3,103])
ax.boxplot(boxStd, 0, '', vert=0, widths=widths, whis=[12.5,87.5])
ax.set_xlim([-3,103])
ax.set_aspect(8)
plt.title('Prediction Errors by Production Anomaly')
plt.yticks(range(1,7), labelsUnsorted)
plt.xlabel('Percent Error from Predicted to Actual Production')
plt.grid(axis='x')
plt.savefig(wdfigs+'box_wisker_anomaly.pdf')
###########################

####### Multivariate #######
exit()
clf=sklearn.linear_model.LinearRegression()
clf.fit(xMulti,ydataMulti)

xMultiNDVI=xMulti[:,0]
ydata=ydataMulti[:]
mask=xMultiNDVI<-0.1
mask[xMultiNDVI>0.5]=1
xMultiNDVI=np.ma.compressed(np.ma.masked_array(xMultiNDVI,mask))
ydata=np.ma.compressed(np.ma.masked_array(ydata,mask))
CorrAfrica=stats.pearsonr(xMultiNDVI,ydata)[0]
slopeA,b=np.polyfit(xMultiNDVI,ydata,1)
yfit=slopeA*xMultiNDVI+b
xMultiNDVI=xMultiNDVI*100

plt.clf()
plt.figure(5,figsize=(3,4))
plt.grid(True)
plt.plot([-30,30],[0,0],'k-',linewidth=2)
plt.plot([0,0],[-3,3],'k-',linewidth=2)
plt.plot(xMultiNDVI,ydata,'b*',xMultiNDVI,yfit,'g-')
plt.xlim([-7,13])
plt.ylim([-.1,.1])
plt.title('NDVI Anom and Corn Yield for every African Country, Corr = '+str(round(CorrAfrica,2)))
plt.xlabel('NDVI*100: All of Africa')
plt.ylabel('Normalized Corn Yield')
plt.savefig(wdfigs+'current_harvest/corrAllAfica.pdf')


exit()

plt.clf()
plt.grid(True)
n, bins, patches = plt.hist(errorAll[errorAll!=0], bins=90, range=(0,90), density=False, facecolor='seagreen',edgecolor='black', alpha=0.9)
plt.xlim([-3,90])
plt.title('Percent Error in 2018 Crop Production Predictions')
plt.ylabel('Number of Predictions')
plt.xlabel('Percent Error from Predicted to Actual Production')
plt.savefig(wdfigs+'current_harvest/error_percent.pdf',dpi=700)
exit()

CorrAll[:,2,:]=-1*CorrAll[:,2,:]
CorrAll[:,5,:]=-1*CorrAll[:,5,:]
plt.clf()
n, bins, patches = plt.hist(np.ma.compressed(CorrsAll), bins=10, range=(0,1),facecolor='blue',edgecolor='black', alpha=.9)
plt.title('Every African Country\'s Max Correlation to Crop Production')
plt.ylabel('Number of Countries')
plt.xlabel('Correlation')
plt.savefig(wdfigs+'Africa_correlations_hist.pdf',dpi=700)

fwrite.close()
fpred.close()
fcorr.close()

#plt.clf()
#plt.plot(ydataNDWI,cropYield[2,:],'b*')
#plt.plot(ydataNDWI,yfit,'g-')
#plt.xlabel('NDWI')
#plt.ylabel('Wheat Yield')
#plt.title('Tunisia: NDWI against Wheat Production, Corr='+str(round(Corr[2,2],2)))
#plt.savefig(wdfigs+'other/Tunisia_NDWI_wheat_corr')

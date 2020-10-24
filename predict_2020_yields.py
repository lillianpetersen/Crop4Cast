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
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

###############################################
# Functions
###############################################
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
fweb = open(wddata+'website_with_error.csv','w')
fcorr = open(wddata+'corrs_every_index_crop_2018'+currentMonth+'.csv','w')

nyearsCrop = 7
nyearsSatellite = 8

monthName=['January','Febuary','March','April','May','June','July','August','September','October','November','December']

# shapes = ncountries, nVIs, ncrops
ncountries = 48
#nVI = 6
nVI = 4
ncrops = 7

slopesAll = np.zeros(shape = (ncountries,nVI,ncrops))
bIntAll = np.zeros(shape = (ncountries,nVI,ncrops))
maxMonthAll = np.zeros(shape = (ncountries))

countryList = np.zeros(shape=(ncountries),dtype='object')
cropAll = np.zeros(shape=(ncountries,ncrops),dtype='object')
predictionsAll = np.zeros(shape=(ncountries,nVI,ncrops))
errorAll = np.zeros(shape=(ncountries,nVI,ncrops))
CorrAll = np.zeros(shape=(ncountries,nVI,ncrops))
GoodCorrAll = np.zeros(shape=(ncountries,nVI,ncrops),dtype='bool') # 1 = Good
predFromStdDevAll = np.zeros(shape=(ncountries,nVI,ncrops))
predPercentAll = np.zeros(shape=(ncountries,nVI,ncrops))
predInPercent = np.zeros(shape=(ncountries,nVI,ncrops))
weightedPredAll = np.zeros(shape=(ncountries))
bestPred = np.zeros(shape=(ncountries,ncrops))
bestPredAnom = np.zeros(shape=(ncountries,ncrops))
countryNamesOriginal = np.zeros(shape=(ncountries),dtype=object)
harvestMonthAll = np.zeros(shape = (ncountries),dtype = int)
harvestMonthAllName = np.zeros(shape=(ncountries),dtype='object')

countryNumsToRun = np.array([2,3,4,5,6,13,24,25,26,27,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
countryNumsToRun = np.array([35])

for icountry in range(1,48):
	countryNum = str(icountry)
	if np.amax(countryNumsToRun==icountry)==0: continue
	
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
	#Good = False
	makePlots = False
	makeFinalPlots = True
	MakePredictions = True
	noCropData = False
			
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

			#if (corrMonthtmp>=2 and corrMonthtmp<6): # July predictions
			##if (corrMonthtmp>5): # Feb predictions
			#	print '\nRunning',country, ' month = '+monthName[corrMonthtmp-1]
			#	Good=True
			#	break
			#else:
			#	print country, 'has other season'
			#	break

	#if Good==False:
	#	continue
	countryNamesOriginal[icountry] = country
		
	####################################################################
	########### load variables ###########
	NDVIanom=np.load(wdvars+country+'/ndviAnomAllYears.npy')
	#EVIanom=np.load(wdvars+country+'/eviAnomAllYears.npy')
	NDWIanom=np.load(wdvars+country+'/ndwiAnomAllYears.npy')
	
	NDVIavg=np.load(wdvars+country+'/ndviAvgAllYears.npy')
	#EVIavg=np.load(wdvars+country+'/eviAvgAllYears.npy')
	NDWIavg=np.load(wdvars+country+'/ndwiAvgAllYears.npy')
	if len(NDVIavg.shape)==3:
		NDVIavg=NDVIavg[0]	
		#EVIavg=EVIavg[0]	
		NDWIavg=NDWIavg[0]	
		NDVIanom=NDVIanom[0]	
		#EVIanom=EVIanom[0]	
		NDWIanom=NDWIanom[0]	
	######################################

	print '\n',country,countryNum

	files = [filename for filename in os.listdir(cropDataDir) if filename.startswith(countryl+'-')]
	crop=[]
	for n in files:
		tmp=n.split('-')
		croptmp=tmp[1].title()
		crop.append(tmp[1].title())
	if len(crop)==0:
		noCropData = True
	
	if noCropData:
		if country=='Libya' or country=='South Sudan':
			VIs = ['NDVI','NDWI']
			yearlyNDVIavg = np.zeros(shape=nyearsSatellite)
			yearlyNDWIavg = np.zeros(shape=nyearsSatellite)
			for y in range(nyearsSatellite):
				yearlyNDVIavg[y] = np.ma.amax(NDVIavg[y])
				yearlyNDWIavg[y] = np.ma.amin(NDWIavg[y])
			for iVI in range(len(VIs)):
				vi = VIs[iVI]
				#vars()['yearly'+vi+'avg'] = np.zeros(shape=nyearsSatellite)
				#vars()['yearly'+vi+'anom'] = np.zeros(shape=nyearsSatellite)
				#for y in range(nyearsSatellite):
				#	vars()['yearly'+vi+'avg'][y] = np.amax(vars()[vi+'avg'][y])
				#	vars()['yearly'+vi+'anom'][y] = vars()[vi+'anom'][y,np.where(vars()[vi+'avg'][y]==vars()['yearly'+vi+'avg'][y])[0][0]]

				## Difference in Percent of VI Averages ##
				meanVI = np.mean(vars()['yearly'+vi+'avg'][:-1])
				percentDiff = (vars()['yearly'+vi+'avg'][-1] - meanVI) / meanVI *100
				predPercentAll[icountry,iVI,0] = percentDiff

				### Difference in Percent of VI Anomalies ##
				#meanVI = np.mean(vars()['yearly'+vi+'anom'][:-1])
				#percentDiff = (vars()['yearly'+vi+'anom'][-1] - meanVI) / meanVI *100
				#predPercentAll[icountry,iVI+2,0] = percentDiff

				GoodCorrAll[icountry,:,0] = 1

		elif country=='Congo':
			viMask = np.sum([NDVIavg==0,np.isnan(NDVIavg)],axis=0,dtype=bool)
			VIs = ['NDVI','NDWI']
			for iVI in range(len(VIs)):
				vi = VIs[iVI]
				vars()[vi+'avg'] = np.ma.masked_array(vars()[vi+'avg'],viMask)
				vars()[vi+'anom'] = np.ma.masked_array(vars()[vi+'anom'],viMask)

				vars()['yearly'+vi+'avg'] = np.zeros(shape=nyearsSatellite)
				vars()['yearly'+vi+'anom'] = np.zeros(shape=nyearsSatellite)
				for y in range(nyearsSatellite):
					ydatatmp1Avg = np.ma.amax(vars()[vi+'avg'][y,corrMonth1-1:corrMonth1+2])
					whereMax1 = np.where(vars()[vi+'avg'][y]==ydatatmp1Avg)[0][0]
					ydatatmp1Anom = np.ma.amax(vars()[vi+'anom'][y,whereMax1])

					ydatatmp2Avg = np.ma.amax(vars()[vi+'avg'][y,corrMonth2-1:corrMonth2+2])
					whereMax2 = np.where(vars()[vi+'avg'][y]==ydatatmp2Avg)[0]
					ydatatmp2Anom = np.ma.amax(vars()[vi+'anom'][y,whereMax2])

					vars()['yearly'+vi+'avg'][y] = np.ma.mean([ydatatmp1Avg,ydatatmp2Avg])
					vars()['yearly'+vi+'anom'][y] = np.ma.mean([ydatatmp1Anom,ydatatmp2Anom])

				if vi=='NDWI':
					vars()['yearly'+vi+'avg'] = vars()['yearly'+vi+'avg']*-1
					vars()['yearly'+vi+'anom'] = vars()['yearly'+vi+'anom']*-1

				## Difference in Percent of VI Averages ##
				meanVI = np.mean(vars()['yearly'+vi+'avg'][:-1])
				percentDiff = (vars()['yearly'+vi+'avg'][-1] - meanVI) / meanVI *100
				predPercentAll[icountry,iVI,0] = percentDiff

				### Difference in Percent of VI Anomalies ##
				#meanVI = np.mean(vars()['yearly'+vi+'anom'][:-1])
				#percentDiff = (vars()['yearly'+vi+'anom'][-1] - meanVI) / meanVI *100
				#predPercentAll[icountry,iVI+2,0] = percentDiff

				GoodCorrAll[icountry,:,0] = 1

		weightedPredAll[icountry] = np.mean(predPercentAll[icountry,:2,0])
		continue
	
	cropYield=np.zeros(shape=(len(files),nyearsCrop))
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
			if year<2013 or year>=2013+nyearsCrop:
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
		elif crop[cp]=='Peanut_Oilseed':
			crop[cp]='Peanuts'
		elif crop[cp]=='Soybean_Meal':
			crop[cp]='Soybeans'
	
	print crop
	
	### Find Crop Yield Anomaly ###
	cropYieldAnom = np.zeros(shape=(cropYield.shape))
	meanYield = np.zeros(shape=(len(crop)))
	for cp in range(len(crop)):
		meanYield[cp]=np.mean(cropYield[cp,:])
		for y in range(nyearsCrop):
			cropYieldAnom[cp,y]=cropYield[cp,y]-meanYield[cp]
	
	if np.ma.amax(cropYieldAnom)==0. and np.amin(cropYieldAnom)==0:
		cropYieldAnom[:]=1
	
	countryList[icountry] = country
	
	if not os.path.exists(wdfigs+country):
		os.makedirs(wdfigs+country)
	
	xtime=np.zeros(shape=(nyearsSatellite,12))
	for y in range(nyearsSatellite):
		for m in range(12):
			xtime[y,m]=(y+2013)+(m+.5)/12
	
	viMask = np.sum([NDVIavg==0,np.isnan(NDVIavg)],axis=0,dtype=bool)
	NDVIavg = np.ma.masked_array(NDVIavg,viMask)
	#EVIavg = np.ma.masked_array(EVIavg,viMask)
	NDWIavg = np.ma.masked_array(NDWIavg,viMask)
	NDVIanom = np.ma.masked_array(NDVIanom,viMask)
	#EVIanom = np.ma.masked_array(EVIanom,viMask)
	NDWIanom = np.ma.masked_array(NDWIanom,viMask)
	xtime = np.ma.masked_array(xtime,viMask)

	NDVIavgPlot=np.ma.compressed(NDVIavg)
	#EVIavgPlot=np.ma.compressed(EVIavg)
	NDWIavgPlot=np.ma.compressed(NDWIavg)
	NDVIanomPlot=np.ma.compressed(NDVIanom)
	#EVIanomPlot=np.ma.compressed(EVIanom)
	NDWIanomPlot=np.ma.compressed(NDWIanom)
	xtime=np.ma.compressed(xtime)

	### Plot NDVI over Time ###
	for cp in range(len(crop)):
		if makePlots:
			plt.close('all')
			plt.clf()
			plt.plot(xtime,NDVIavgPlot,'-*b')
			plt.ylabel('NDVI Monthly Average')
			plt.title(country+' NDVI Monthly Average')
			plt.grid(True)
			plt.savefig(wdfigs+country+'/NDVIavg_over_time.pdf',dpi=700)
			plt.clf()
	
	########################
	Corr = np.zeros(shape=(nVI,ncrops))
	slope = np.zeros(shape=(nVI,ncrops))
	bInt = np.zeros(shape=(nVI,ncrops))
	data2017 = np.zeros(shape=(nVI,ncrops))
	
	###########################################
	# Monthly VI Avg and Crop Yield
	###########################################
	bar_width = 0.2

	#VIs = ['NDVI','EVI','NDWI']
	VIs = ['NDVI','NDWI']
	
	for iVI in range(len(VIs)):
		vi = VIs[iVI]
		## Find Max VIs for each year ##
		vars()['max'+vi+'avg'] = np.zeros(shape=(nyearsSatellite))
		vars()['max'+vi+'anom'] = np.zeros(shape=(nyearsSatellite))
		maxMonth=np.zeros(shape=(nyearsSatellite),dtype=int)
		if twoSeasons!='no':
			for y in range(nyearsSatellite):
				ydatatmp1 = np.ma.amax(vars()[vi+'avg'][y,corrMonth1-1:corrMonth1+2])
				if country=='Rwanda':
					if y!=0:
						ydatatmp2=np.ma.amax(vars()[vi+'avg'][y-1,corrMonth2-1:corrMonth2+2])
					else:
						ydatatmp2=ydatatmp1
				else:
					ydatatmp2=np.ma.amax(vars()[vi+'avg'][y,corrMonth2-1:corrMonth2+2])
				vars()['max'+vi+'avg'][y]=np.ma.mean([ydatatmp1,ydatatmp2])
		else:
			for y in range(nyearsSatellite):
				vars()['max'+vi+'avg'][y] = np.ma.amax(vars()[vi+'avg'][y,:])
				maxMonth[y] = np.ma.where(vars()[vi+'avg'][y,:]==np.ma.amax(vars()[vi+'avg'][y,:]))[0][0]
			# If VI max may be over year boundary
			if np.any(maxMonth==10) or np.any(maxMonth==11) or np.any(maxMonth==0) or np.any(maxMonth==1):
				SeasonOverYear = True
				maxMonthWYears = np.zeros(shape=(nyearsSatellite,2),dtype=int)
				maxtmp = np.ma.amax(vars()[vi+'avg'][0,:6])
				maxMonthWYears[0,:] = np.ma.where(vars()[vi+'avg'][:,:]==maxtmp)[0][0],np.ma.where(vars()[vi+'avg'][:,:]==maxtmp)[1][0]
				for y in range(1,nyearsSatellite):
					maxtmp1 = np.ma.amax(vars()[vi+'avg'][y-1,6:])
					maxtmp2 = np.ma.amax(vars()[vi+'avg'][y,:6])
					maxtmp = np.ma.amax([maxtmp1,maxtmp2])
					maxMonthWYears[y,:] = np.ma.where(vars()[vi+'avg'][:,:]==maxtmp)[0][0],np.ma.where(vars()[vi+'avg'][:,:]==maxtmp)[1][0]

			if SeasonOverYear:
				for y in range(nyearsSatellite):
					vars()['max'+vi+'avg'][y] = vars()[vi+'avg'][maxMonthWYears[y,0],maxMonthWYears[y,1]]

		## Mask the index ##
		wherenan = np.where(np.isnan(vars()['max'+vi+'avg'])==True)
		masktmp = np.zeros(shape=5)
		masktmp[wherenan] = 1
		if np.sum(masktmp)>0:
			for y in range(nyearsSatellite):
				vars()['max'+vi+'avg'][y] = np.amax(vars()[vi+'avg'][y])
		####################

		## Plot VI line with Crop bar ##
		for cp in range(len(crop)):
			if corrMonth != 'Max':
				maxMonth[:] = float(corrMonth)
				for y in range(nyearsSatellite):
					vars()['max'+vi+'avg'][y] = vars()[vi+'avg'][y,int(corrMonth)]
			if corrYear == 'after':
				Corr[iVI,cp] = stats.pearsonr(cropYield[cp,1:],vars()['max'+vi+'avg'][:nyearsCrop-1])[0]
				xCrop = np.arange(2013.+(harvestMonth/12)+1,2013.+(harvestMonth/12)+nyearsCrop)
				cropData = cropYield[cp,1:]
				slope[iVI,cp],bInt[iVI,cp] = np.polyfit(vars()['max'+vi+'avg'][:nyearsCrop-1],cropYield[cp,1:],1)
				yfit = slope[iVI,cp]*vars()['max'+vi+'avg'][:-1]+bInt[iVI,cp]
			elif corrYear == 'same':
				Corr[iVI,cp] = stats.pearsonr(cropYield[cp,:nyearsCrop],vars()['max'+vi+'avg'][:nyearsCrop])[0]
				xCrop = np.arange(2013.+(harvestMonth/12),2013.+(harvestMonth/12)+nyearsCrop)
				cropData = cropYield[cp,:]
				slope[iVI,cp],bInt[iVI,cp] = np.polyfit(vars()['max'+vi+'avg'][:nyearsCrop],cropYield[cp,:nyearsCrop],1)
				yfit = slope[iVI,cp]*vars()['max'+vi+'avg'][:nyearsCrop]+bInt[iVI,cp]
			
			if makePlots:
				plt.clf()
				fig, ax2 = plt.subplots()
				ax1 = ax2.twinx()
				ax2.grid(True)
		
				label=crop[cp]+' Production'
				ax2.bar(xCrop,cropData,bar_width,color='g',label=label)
				ax2.legend(loc='upper right')
				ax2.tick_params(axis='y',colors='g')
				ax2.set_ylabel(crop[cp]+' Production, Gigatonnes',color='g')
		
				ax2.set_ylim([np.ma.amin(cropData)*.06,np.ma.amax(cropData)*1.15])
				ax1.set_ylim([np.ma.amin(vars()[vi+'avgPlot'])*100*.9,np.ma.amax(vars()[vi+'avgPlot'])*100*1.1])
		
				ax1.plot(xtime,vars()[vi+'avgPlot']*100,'b*-',label='Monthly '+vi)
				ax1.legend(loc='upper left')
				ax1.set_ylabel(vi+' Monthly Average * 100',color='b')
				ax1.tick_params(axis='y',colors='b')
		
				plt.title(country+': '+vi+' Monthly Average and '+crop[cp]+' Production, Corr='+str(round(Corr[iVI,cp],2)))
				plt.savefig(wdfigs+country+'/monthly_'+vi+'_avg_with_'+crop[cp]+'.pdf',dpi=700)

	###########################################
	# Yearly VI Avg and Crop Yield
	###########################################
	bar_width = 0.27

	for iVI in range(len(VIs)):
		vi = VIs[iVI]
		if vi=='EVI': continue

		for cp in range(len(crop)):
			cropData = cropYield[cp,:]
			xCrop = np.arange(2013+.14,2013+nyearsCrop+.14)
			xVI = np.arange(2013-.14,2013+nyearsSatellite-.14)
		
			if makePlots:
				plt.clf()
				fig, ax2 = plt.subplots()
				ax1 = ax2.twinx()
				
				ax1.set_ylim([np.ma.amin(cropData)*0.9,np.ma.amax(cropData)*1.05])
				ax2.set_ylim([np.ma.amin(vars()['max'+vi+'avg']*100)*0.9,np.ma.amax(vars()['max'+vi+'avg']*100)*1.05])
		
				ax2.set_ylabel(vi+' Monthly Average * 100',color='b')
				label=crop[cp]+' Production'
				ax1.bar(xCrop,cropData,bar_width,color='g',label=label)
				ax1.tick_params(axis='y',colors='g')
				ax2.tick_params(axis='y',colors='b')
				if vi=='NDWI':
					ax2.bar(xVI,vars()['max'+vi+'avg']*100-100,bar_width,color='b',bottom=100,label='Max '+vi)
				else:
					ax2.bar(xVI,vars()['max'+vi+'avg']*100,bar_width,color='b',label='Max '+vi)
		
				ax1.set_ylabel('Production, Gigatonnes',color='g')
				plt.title(country+': '+corrMonthName+' '+vi+' and '+crop[cp]+' Production, Corr='+str(round(Corr[iVI,cp],2)))
				ax2.grid(True)
				ax1.set_xticks(range(2013,2013+nyearsSatellite))
				ax1.legend(loc='upper right')
				ax2.legend(loc='upper left')
				
				if vi=='NDWI':
					ax2.set_ylim(ax2.get_ylim()[::-1])

				plt.savefig(wdfigs+country+'/'+vi+'_avg_'+crop[cp]+'_'+country+'.pdf')
	
	###########################################
	# Yearly VI Anom and Crop Yield
	###########################################
	bar_width = 0.27

	for iVI in range(len(VIs)):
		vi = VIs[iVI]
		if vi=='EVI': continue

		for cp in range(len(crop)):
			cropData = cropYieldAnom[cp,:]
			xCrop = np.arange(2013+.14,2013+nyearsCrop+.14)
			xVI = np.arange(2013-.14,2013+nyearsSatellite-.14)
			if twoSeasons!='no':
				for y in range(nyearsSatellite):
					ydatatmp1Avg = np.ma.amax(vars()[vi+'avg'][y,corrMonth1-1:corrMonth1+2])
					whereMax1 = np.where(vars()[vi+'avg'][y]==ydatatmp1Avg)[0][0]
					ydatatmp1 = np.ma.amax(vars()[vi+'anom'][y,whereMax1])
					if country=='Rwanda':
						if y!=0:
							ydatatmp2 = np.ma.amax(vars()[vi+'anom'][y-1,corrMonth2-1:corrMonth2+2])
						else:
							ydatatmp2 = ydatatmp1
					else:
						ydatatmp2Avg = np.ma.amax(vars()[vi+'avg'][y,corrMonth2-1:corrMonth2+2])
						whereMax2 = np.where(vars()[vi+'avg'][y]==ydatatmp2Avg)[0][0]
						ydatatmp2 = np.ma.amax(vars()[vi+'anom'][y,whereMax2])
					vars()['max'+vi+'anom'][y] = np.ma.mean([ydatatmp1,ydatatmp2])
			else:
				for y in range(nyearsSatellite):
					vars()['max'+vi+'anom'][y] = vars()[vi+'anom'][y,maxMonth[y]]
					if SeasonOverYear:
						vars()['max'+vi+'anom'][y] = vars()[vi+'anom'][maxMonthWYears[y,0],maxMonthWYears[y,1]]

			## Mask the index ##
			wherenan = np.where(np.isnan(vars()['max'+vi+'anom'])==True)
			masktmp = np.zeros(shape=5)
			masktmp[wherenan] = 1
			if np.sum(masktmp)>0:
				for y in range(nyearsSatellite):
					vars()['max'+vi+'anom'][y] = np.amax(vars()[vi+'anom'][y])
			####################

			if corrMonth!='Max':
				vars()['max'+vi+'anom'] = vars()[vi+'anom'][:,int(corrMonth)]

			if corrYear=='after':
				cropData = cropYieldAnom[cp,1:]
				Corr[iVI+2,cp] = stats.pearsonr(cropYieldAnom[cp,1:],vars()['max'+vi+'anom'][:nyearsCrop-1])[0]
				slope[iVI+2,cp],bInt[iVI+2,cp] = np.polyfit(vars()['max'+vi+'anom'][:nyearsCrop-1],cropYield[cp,1:nyearsCrop],1)
				yfit = slope[iVI+2,cp]*vars()['max'+vi+'anom'][:nyearsCrop-1]+bInt[iVI+2,cp]

			elif corrYear=='same':
				Corr[iVI+2,cp] = stats.pearsonr(cropYieldAnom[cp,:nyearsCrop],vars()['max'+vi+'anom'][:nyearsCrop])[0]
				slope[iVI+2,cp],bInt[iVI+2,cp] = np.polyfit(vars()['max'+vi+'anom'][:nyearsCrop],cropYield[cp,:nyearsCrop],1)
				yfit = slope[iVI+2,cp]*vars()['max'+vi+'anom'][:]+bInt[iVI+2,cp]
		
			if makePlots:
				plt.clf()
				fig, ax2 = plt.subplots()
				ax1 = ax2.twinx()
				
				if vi=='NDWI':
					ax1.set_ylim([-1*np.ma.amax(abs(cropData))*1.05,np.ma.amax(abs(cropData))*1.2])
					ax2.set_ylim([-1*np.ma.amax(abs(vars()['max'+vi+'anom']))*100*1.2,np.ma.amax(abs(vars()['max'+vi+'anom']))*100*1.05])
				else:
					ax1.set_ylim([-1*np.ma.amax(abs(cropData))*1.05,np.ma.amax(abs(cropData))*1.2])
					ax2.set_ylim([-1*np.ma.amax(abs(vars()['max'+vi+'anom']))*100*1.05,np.ma.amax(abs(vars()['max'+vi+'anom']))*100*1.2])
		
				ax2.set_ylabel(vi+' Anomaly * 100',color='b')
				label = crop[cp]+' Production'
				if corrYear=='same':
					ax1.bar(xCrop,cropData,bar_width,color='g',label=label)
				elif corrYear=='after':
					ax1.bar(xCrop[1:],cropData,bar_width,color='g',label=label)
				ax1.tick_params(axis='y',colors='g')
				ax2.tick_params(axis='y',colors='b')
				ax2.bar(xVI,vars()['max'+vi+'anom']*100,bar_width,color='b',label=vi+' Anom')
		
				ax1.set_ylabel('Production Anomaly, Gigatonnes',color='g')
				plt.title(country+': '+vi+' Anom and '+crop[cp]+' Production, Corr='+str(round(Corr[iVI+2,cp],2)))
				ax2.grid(True)
				ax1.set_xticks(range(2013,2013+nyearsSatellite))
				ax1.legend(loc='upper right')
				ax2.legend(loc='upper left')

				if vi=='NDWI':
					ax2.set_ylim(ax2.get_ylim()[::-1])

				plt.savefig(wdfigs+country+'/'+vi+'_anom_'+crop[cp]+'_'+country+'.pdf',dpi=700)
	
	variables = ['NDVIavg','NDWIavg','NDVIanom','NDWIanom']
	
	## Find Max Corr, Write to General Arrays ##
	Corr = np.ma.masked_array(Corr,Corr==0)
	maxCorr = np.ma.amax(Corr)

	whereMaxCorrX = np.where(Corr==maxCorr)[0][0]
	whereMaxCorrY = np.where(Corr==maxCorr)[1][0]

	slopesAll[icountry] = slope[:,:]
	bIntAll[icountry] = bInt[:,:]
	############################################

	## Find max NDVI month and estimate harvest month ##
	data = Counter(maxMonth)
	common = data.most_common(1)[0][0]
	maxMonthAll[icountry] = common
	harvestMonthAll[icountry] = maxMonthAll[icountry]+3
	seasonHight = int(maxMonthAll[icountry])
	if harvestMonthAll[icountry]>11:
		harvestMonthAll[icountry] = harvestMonthAll[icountry]-12
	harvestMonthAll[0] = 11
	####################################################

	print 'Max Corr =', round(maxCorr,2)
	print 'All Corrs =','\n',np.round(Corr,2).data
	print 'Max Corr Crop = '+crop[whereMaxCorrY]+', Var = '+variables[whereMaxCorrX]
	print 'Max Corr Month =', monthName[int(maxMonthAll[icountry])]

	cropAll[icountry,:len(crop)] = crop
	CorrAll[icountry] = Corr
	for iVI in range(len(VIs)):
		vi = VIs[iVI]
		for cp in range(len(crop)):
			if CorrAll[icountry,iVI,cp]>0.3 and iVI==0:
				GoodCorrAll[icountry,iVI,cp] = 1 # set to good
			if CorrAll[icountry,iVI,cp]<-0.3 and iVI==1:
				GoodCorrAll[icountry,iVI,cp] = 1 # set to good
			if CorrAll[icountry,iVI+2,cp]>0.3 and iVI==0:
				GoodCorrAll[icountry,iVI+2,cp] = 1 # set to good
			if CorrAll[icountry,iVI+2,cp]<-0.3 and iVI==1:
				GoodCorrAll[icountry,iVI+2,cp] = 1 # set to good
	
	if maxCorr<.7:
		print 'low corr'

	if MakePredictions==False:
		print 'MakePredictions = False'
		continue

	#############################################################################
	# Make Predictions
	#############################################################################
	if not MakePredictions: continue

	bar_width = 0.2
	for iVI in range(len(VIs)):
		vi = VIs[iVI]
		if vi=='EVI': continue

		for cp in range(len(crop)):

			###########################################################################
			# Averages
			###########################################################################

			#### Predict 2020 yield ###
			yieldPred = slope[iVI,cp] * vars()['max'+vi+'avg'][-1] + bInt[iVI,cp]
			predAnom = yieldPred - meanYield[cp]
			predPercent = (yieldPred - meanYield[cp]) / meanYield[cp] *100
	
			predictionsAll[icountry,iVI,cp] = yieldPred
			predFromStdDevAll[icountry,iVI,cp] = predAnom
			predPercentAll[icountry,iVI,cp] = predPercent
			###########################

			viMonthly = np.ma.compressed(vars()[vi+'avg'])
			viYearly = vars()['max'+vi+'avg']

			###########################################
			# Monthly VI with Prediction
			###########################################
	
			if makeFinalPlots:
				props = dict(boxstyle='round', facecolor='white', alpha=1)
	
				plt.clf()
				fig, ax1 = plt.subplots()
				ax2 = ax1.twinx()
				ax3 = ax1.twinx()
				ax1.grid(True)
				
				xCrop = np.arange(2013.+(harvestMonth/12),2013.+nyearsCrop+(harvestMonth/12))
				cropData = cropYield[cp]
	
				ax2.bar(xCrop,cropData,bar_width,color='g',label=crop[cp]+' Production')
				ax2.bar(xCrop[-1]+1,yieldPred,bar_width,color='m',label='Predicted Production')
	
				ax2.legend(loc='upper right')
				ax2.tick_params(axis='y',colors='g')
				ax2.set_ylabel(crop[cp]+' Production, Gigatonnes',color='g')
				
				ax2.set_ylim([np.ma.amin(cropData)*.96,np.amax([np.amax(cropData),np.amax(yieldPred)])*1.02])
				ax1.set_ylim([np.ma.amin(viMonthly*100)*.9,np.amax( [np.amax(viMonthly*100)] )*1.15])
				
				ax1.plot(xtime,viMonthly*100,'b*-',label='Monthly '+vi)
				ax1.legend(loc='upper left')
				
				ax1.set_ylabel(vi+' Monthly Average * 100',color='b')
				ax1.tick_params(axis='y',colors='b')
	
				ax3.text(2015.9, .93, 'Corr='+str(round(Corr[iVI,cp],2)),bbox=props)
				ax3.axis('off')
				
				if predPercent>0:
					plt.title(country+': '+vi+' and '+crop[cp]+', Pred = '+str(abs(int(predPercent)))+'% Above Avg')
				elif predPercent<0:
					plt.title(country+': '+vi+' and '+crop[cp]+', Pred = '+str(abs(int(predPercent)))+'% Below Avg')
				plt.savefig(wdfigs+country+'/'+country+'_prediction_'+vi+'_monthly_with_'+crop[cp]+'.pdf',dpi=700)
			
			###########################################
			# Yearly VI with Prediction
			###########################################

			if makeFinalPlots:
				bar_width = 0.27
				
				xVI = np.arange(2013-.14,2013+nyearsSatellite-.14)
				xCrop = np.arange(2013+.14,2013+nyearsCrop+.14)
				cropData = cropYield[cp]
				
				plt.clf()
				fig, ax2 = plt.subplots()
				ax1 = ax2.twinx()
				ax3 = ax2.twinx()
				
				ax1.set_ylim([np.ma.amin([np.ma.amin(cropData),yieldPred])*.4,np.ma.amax([np.ma.amax(cropData),yieldPred])*1.3]) 
				if vi=='NDWI' and np.amin(viYearly)<0:
					ax2.set_ylim([np.ma.amin(viYearly*100)*1.1,np.ma.amax(viYearly*100)*0.9])
				else:
					ax2.set_ylim([np.ma.amin(viYearly*100)*0.98,np.ma.amax(viYearly*100)*1.02])
				
				if vi=='NDWI':
					ax2.set_ylabel('Max '+vi+' Month Average *100, flipped',color='b')
				else:
					ax2.set_ylabel('Max '+vi+' Month Average *100',color='b')
				
				label = crop[cp]+' Production'
				#if corrYear=='after':
				#	print 'corrYear after'
				#	exit()
				#	ax1.bar(xCrop[:-1],cropData,bar_width,color='g',label=label)
				#	ax1.bar(2019.14,yieldPred,bar_width,color='m',label='Predicted Production')
				#	label='Predicted production'
				#	#ax1.bar(xCrop[-1],yieldPred,bar_width,color='m',label=label)
				#	ax1.bar(2017.14,crop2018[cp],bar_width,color='g')
				#else:

				ax1.bar(xCrop,cropData,bar_width,color='g',label=crop[cp]+' Production')
				ax1.bar(xCrop[-1]+1,yieldPred,bar_width,color='m',label='Predicted Production')

				ax1.tick_params(axis='y',colors='g')
				ax2.tick_params(axis='y',colors='b')

				if vi=='NDWI':
					ax2.bar(xVI,viYearly*100-100,bar_width,color='b',bottom=100,label='Max '+vi)
				else:
					ax2.bar(xVI,(viYearly*100),bar_width,color='b',label='Max '+vi)
	
				# Corr text
				ax3.text(2015.6, .93, 'Corr='+str(round(Corr[iVI,cp],2)),bbox=props)
				
				ax1.set_ylabel('Production, Gigatonnes',color='g')
				ax3.axis('off')
				if predPercent>0:
					plt.title(country+': Max '+vi+' and '+crop[cp]+', Prediction = '+str(abs(int(predPercent)))+'% Above Avg')
				elif predPercent<0:
					plt.title(country+': Max '+vi+' and '+crop[cp]+', Prediction = '+str(abs(int(predPercent)))+'% Below Avg')
				ax2.grid(True)
				ax1.set_xticks(range(2013,2013+nyearsSatellite))
				ax2.set_xticks(range(2013,2013+nyearsSatellite))
				ax1.legend(loc='upper right')
				ax2.legend(loc='upper left')

				if vi=='NDWI':
					ax2.set_ylim(ax2.get_ylim()[::-1])

				plt.savefig(wdfigs+country+'/'+country+'_prediction_'+vi+'_with_'+crop[cp]+'.pdf')
				if vi=='NDVI' and Corr[iVI,cp]>0.6:
					plt.savefig(wdfigs+'for_prediction/'+country+'_'+vi+'_avg_with_'+crop[cp]+'.pdf')
				if vi=='NDWI' and Corr[iVI,cp]<-0.6:
					plt.savefig(wdfigs+'for_prediction/'+country+'_'+vi+'_avg_with_'+crop[cp]+'.pdf')
			
			###########################################################################
			# Anomalies
			###########################################################################
			
			#### Predict 2020 yield ###
			yieldPred = slope[iVI+2,cp] * vars()['max'+vi+'anom'][-1] + bInt[iVI+2,cp]
			predAnom = yieldPred - meanYield[cp]
			predPercent = (yieldPred - meanYield[cp]) / meanYield[cp] *100

			predictionsAll[icountry,iVI+2,cp] = yieldPred
			predFromStdDevAll[icountry,iVI+2,cp] = predAnom
			predPercentAll[icountry,iVI+2,cp] = predPercent
			###########################

			viYearly = vars()['max'+vi+'anom']

			###########################################
			# Yearly VI with Prediction
			###########################################
	
			if makeFinalPlots:
				bar_width = 0.27
				
				xVI = np.arange(2013-.14,2013+nyearsSatellite-.14)
				xCrop = np.arange(2013+.14,2013+nyearsCrop+.14)
				cropData = cropYieldAnom[cp]
				
				plt.clf()
				fig, ax2 = plt.subplots()
				ax1 = ax2.twinx()
				ax3 = ax2.twinx()
				
				if vi=='NDWI':
					ax2.set_ylabel(vi+' Anomaly *100, flipped',color='b')
					ax1.set_ylim([np.amax([np.amax(np.abs(cropData)),np.abs(yieldPred)])*-1.2, np.amax([np.amax(np.abs(cropData)),np.abs(yieldPred)])*1.5]) 
					ax2.set_ylim([np.amax(np.abs(viYearly*100))*-1.5, np.amax(np.abs(viYearly*100))*1.2])
				else:
					ax2.set_ylabel(vi+' Anomaly *100, flipped',color='b')
					ax1.set_ylim([np.amax([np.amax(np.abs(cropData)),np.abs(yieldPred)])*-1.2, np.amax([np.amax(np.abs(cropData)),np.abs(yieldPred)])*1.5]) 
					ax2.set_ylim([np.amax(np.abs(viYearly*100))*-1.2, np.amax(np.abs(viYearly*100))*1.5])
	
				#if corrYear=='after':
				#	print 'corrYear = after'
				#	exit()
				#	ax1.bar(xCrop[:-1],cropData+1000,bar_width,bottom=-1000,color='g',label=crop[cp]+' Production')
				#	ax1.bar(2017.14,crop2018Anom[cp]+1000,bar_width,bottom=-1000,color='g')
				#	ax1.bar(2019.14,predAnom+1000,bar_width,bottom=-1000,color='g')
				#else:

				ax1.bar(xCrop,cropData,bar_width,color='g',label=crop[cp]+' Production')
				ax1.bar(xCrop[-1]+1,predAnom,bar_width,color='m',label='Predicted Production')

				ax1.tick_params(axis='y',colors='g')
				ax2.tick_params(axis='y',colors='b')

				ax2.bar(xVI,viYearly*100,bar_width,color='b',label=vi+' Anom')
				
				# Corr text
				props = dict(boxstyle='round', facecolor='white', alpha=1)
				ax3.text(2015.6, .93, 'Corr='+str(round(Corr[iVI+2,cp],2)),bbox=props)
				
				ax3.axis('off')
				ax1.set_ylabel('Production Anomaly, Gigatonnes',color='g')
				if predPercent>0:
					plt.title(country+': '+vi+' Anom and '+crop[cp]+', Prediction = '+str(abs(int(predPercent)))+'% Above Avg')
				elif predPercent<0:
					plt.title(country+': '+vi+' Anom and '+crop[cp]+', Prediction = '+str(abs(int(predPercent)))+'% Below Avg')
				ax2.grid(True)

				ax1.set_xticks(range(2013,2013+nyearsSatellite))

				ax1.legend(loc='upper right')
				ax2.legend(loc='upper left')

				if vi=='NDWI':
					ax2.set_ylim(ax2.get_ylim()[::-1])

				plt.savefig(wdfigs+country+'/'+country+'_prediction_'+vi+'_anom_with_'+crop[cp]+'.pdf')
				if vi=='NDVI' and Corr[iVI+2,cp]>0.6:
					plt.savefig(wdfigs+'for_prediction/'+country+'_'+vi+'_anom_with_'+crop[cp]+'.pdf')
				if vi=='NDWI' and Corr[iVI+2,cp]<-0.6:
					plt.savefig(wdfigs+'for_prediction/'+country+'_'+vi+'_anom_with_'+crop[cp]+'.pdf')


	goodVIs = np.amax(GoodCorrAll[icountry],axis=1)
	weightedPred = 0
	for iVI in np.arange(4)[goodVIs]:
		predThisVI = np.mean(predPercentAll[icountry,iVI][GoodCorrAll[icountry,iVI]])
		weightedPred += predThisVI
	weightedPredAll[icountry] = weightedPred/np.sum(goodVIs)
###################################################

exit()
sort = weightedPredAll.argsort()
weightedPred= np.array(weightedPredAll)[sort]
countryNames = np.array(countryNamesOriginal)[sort]
countryNames[countryNames=='Central Africa Republic'] = 'Cen. African Rep.'
countryNames = countryNames[weightedPred!=0]
weightedPred = weightedPred[weightedPred!=0]

color = np.zeros(shape=len(countryNames),dtype=object)
color[:] = 'blue'
color[weightedPred<-15] = 'red'
color[weightedPred>15] = 'green'

plt.close()
plt.clf()
fig, ax = plt.subplots()
ax.barh(np.arange(len(countryNames)), width=weightedPred[::-1], height=0.8, color=color[::-1],zorder=3)
ax.set_yticks(np.arange(-0.5,len(countryNames)+0.5), minor=True)
ax.set_xlabel('Predicted Production Anomaly, % Difference from Average')
ax.grid(which='minor', zorder=0)
ax.plot([0,0],[-5,len(countryNames)+5],'k-',linewidth=3,zorder=3)
ax.set_xlim([-60,60])
ax.set_ylim([-1,len(countryNames)])
plt.gcf().subplots_adjust(left=0.21, right=0.96)
plt.title('Crop4cast Predictions, updated July 2020',fontsize=17)
plt.yticks(np.arange(0,len(countryNames)),countryNames[::-1])
plt.savefig(wdfigs+'crop4cast_predicted_yields_july.pdf')
plt.savefig(wdfigs+'crop4cast_predicted_yields_july.jpg',dpi=200)
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
cmin = np.amin(weightedPred)
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
	xCrop=factoryPctOne[0,c]
	y=y1+(y2-y1)/(cmax-cmin)*(xCrop-cmin)
	icmap=min(255,int(round(y,1)))
	icmap=max(0,int(round(icmap,1)))

	if xCrop!=0:
		size = 15*(1+xCrop/cmax)
		plt.plot(capitalLatLon[1,c], capitalLatLon[0,c], marker='*', markersize=size, color=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]], markeredgewidth=1.5, markeredgecolor='k')
		factoryNumOne+=1
	if xCrop==0:
		plt.plot(capitalLatLon[1,c], capitalLatLon[0,c], marker='*', markersize=7, color='darkred')


for icoast in range(9,len(countrycosted)):
	xCrop=factoryPctOne[0,icoast]
	y=y1+(y2-y1)/(cmax-cmin)*(xCrop-cmin)
	icmap=min(255,int(round(y,1)))
	icmap=max(0,int(round(icmap,1)))
	if xCrop!=0:
		size = 15*(0.8+xCrop/cmax)
		plt.plot(capitalLatLon[1,icoast], capitalLatLon[0,icoast], marker='o', markersize=size, color=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]], markeredgewidth=1.5, markeredgecolor='k')
		IntlNumOne+=1
	if xCrop==0:
		plt.plot(capitalLatLon[1,icoast], capitalLatLon[0,icoast], marker='o', markersize=7, color='darkred')

local = str(int(np.round(100 * np.sum(factoryPctOne[0,:9]) / np.sum(factoryPctOne[0,:]),0)))
intl = str(int(np.round(100 * np.sum(factoryPctOne[0,9:]) / np.sum(factoryPctOne[0,:]),0)))

plt.title('Production of Treatment by Factory and Port', fontsize=18)
plt.legend(loc = 'lower left')
plt.text(-15,-10,str(factoryNumOne)+' Factories Open\n'+str(IntlNumOne)+' Ports Open\n'+local+'% Produced Locally', bbox=dict(fc="none", boxstyle="round"), size = 10)

plt.savefig(wdfigs+'current_factories_demand/'+Ltitles[L]+'/geographical/Export_map.pdf')


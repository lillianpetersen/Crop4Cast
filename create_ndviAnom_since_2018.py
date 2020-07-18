import os
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from sys import exit

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
#my_cmap_r=make_cmap(colors[::-1])

colors = [(128, 66, 0), (255, 230, 204), (255,255,255), (204, 255, 204), (0,100,0)]
my_cmap_gwb = make_cmap(colors,bit=True)
#my_cmap_gwb_r=make_cmap(colors[::-1],bit=True)


wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/science_fair_2018/data/'
wdvars='/Users/lilllianpetersen/science_fair_2018/saved_vars/'
wdfigs='/Users/lilllianpetersen/science_fair_2018/figures/'

countylats=np.load(wdvars+'county_lats.npy')
countylons=np.load(wdvars+'county_lons.npy')
countyName=np.load(wdvars+'countyName.npy')
stateName=np.load(wdvars+'stateName.npy')

recentSeason = False
since2018 = True

startMonth=0
endMonth=12
nmonths=12
nyears=2

makePlots=False

monthName=['January','Febuary','March','April','May','June','July','August','September','October','November','December']

for icountry in range(47):
	if icountry==0:
		continue
	if icountry!=1: continue # Ethiopia

	Good=False

	########### Get Country Information ###########
	fseason=open(wddata+'max_ndviMonths_final.csv','r')
	for line in fseason:
		tmp=line.split(',')
		if tmp[0]==str(icountry):
			country=tmp[1]
			sName=country

			corrMonth=tmp[2][:-1]
			if len(corrMonth)>2:
				months=corrMonth.split('/')
				month1=corrMonth[0]
				month2=corrMonth[1]
				corrMonth=month1
			corrMonth=int(corrMonth)+1
			break

	print sName
	###############################################

	'''
	########### load vars ###########
	climoCounterAll = np.load(wdvars+sName+'/since2018/climoCounterUnprocessed.npy')
	ndviMonthAvgU = np.load(wdvars+sName+'/since2018/ndviMonthAvgUnprocessed.npy')
	eviMonthAvgU = np.load(wdvars+sName+'/since2018/eviMonthAvgUnprocessed.npy')
	ndwiMonthAvgU = np.load(wdvars+sName+'/since2018/ndwiMonthAvgUnprocessed.npy')
	npixels = climoCounterAll.shape[-1]
	######################################
	'''
	ndviMonthAvg = np.load(wdvars+sName+'/ndviMonthAvg.npy')[:5]
	eviMonthAvg = np.load(wdvars+sName+'/eviMonthAvg.npy')[:5]
	ndwiMonthAvg = np.load(wdvars+sName+'/ndwiMonthAvg.npy')[:5]

	########### Load Monthly Climatologies ###########
	ndviClimo = np.load(wdvars+sName+'/ndviClimo.npy')
	eviClimo = np.load(wdvars+sName+'/eviClimo.npy')
	ndwiClimo = np.load(wdvars+sName+'/ndwiClimo.npy')
	##################################################

	########### Compute Pixel-wise Averages and Anomalies ###########
	'''
	ndviMonthAvg = np.nan_to_num(ndviMonthAvgU/climoCounterAll)
	eviMonthAvg = np.nan_to_num(eviMonthAvgU/climoCounterAll)
	ndwiMonthAvg = np.nan_to_num(ndwiMonthAvgU/climoCounterAll)
	'''

	ndviMonthAvg = np.nan_to_num(ndviMonthAvg)
	
	ndviMonthAvg = np.ma.masked_array(ndviMonthAvg,ndviMonthAvg==0)
	eviMonthAvg = np.ma.masked_array(eviMonthAvg,eviMonthAvg==0)
	ndwiMonthAvg = np.ma.masked_array(ndwiMonthAvg,ndwiMonthAvg==0)

	ndviAnomAllPix = np.nan_to_num(ndviMonthAvg - ndviClimo)
	eviAnomAllPix = np.nan_to_num(eviMonthAvg - eviClimo)
	ndwiAnomAllPix = np.nan_to_num(ndwiMonthAvg - ndwiClimo)
	#################################################################

	########### Compute Composite Anomalies and Avgs ###########
	exit()
	ndviAnomCurrent = np.ma.mean(np.ma.mean(ndviAnomAllPix,axis=2),axis=2)
	eviAnomCurrent = np.ma.mean(np.ma.mean(eviAnomAllPix,axis=2),axis=2)
	ndwiAnomCurrent = np.ma.mean(np.ma.mean(ndwiAnomAllPix,axis=2),axis=2)

	#ndviAvgCurrent = np.ma.mean(np.ma.mean(ndviMonthAvg,axis=2),axis=2)
	#eviAvgCurrent = np.ma.mean(np.ma.mean(eviMonthAvg,axis=2),axis=2)
	#ndwiAvgCurrent = np.ma.mean(np.ma.mean(ndwiMonthAvg,axis=2),axis=2)

	ndviAvg = np.ma.mean(np.ma.reshape(ndviMonthAvg, [5,12,2048**2]),axis=2)

	print ndviAnomCurrent[:],'\n'
	print ndviAvgCurrent[:]
	
	np.save(wdvars+sName+'/since2018/ndviAnom',np.array(ndviAnomCurrent))
	np.save(wdvars+sName+'/since2018/eviAnom',np.array(eviAnomCurrent))
	np.save(wdvars+sName+'/since2018/ndwiAnom',np.array(ndwiAnomCurrent))
		
	np.save(wdvars+sName+'/since2018/ndviAvg',np.array(ndviAvgCurrent))
	np.save(wdvars+sName+'/since2018/eviAvg',np.array(eviAvgCurrent))
	np.save(wdvars+sName+'/since2018/ndwiAvg',np.array(ndwiAvgCurrent))
	############################################################
	
	########### Make One Array for 2013-2020 ###########
	ndviAvg2013to17 = np.load(wdvars+sName+'/ndviAvg.npy')[0]
	eviAvg2013to17 = np.load(wdvars+sName+'/eviAvg.npy')[0]
	ndwiAvg2013to17 = np.load(wdvars+sName+'/ndwiAvg.npy')[0]

	ndviAnom2013to17 = np.load(wdvars+sName+'/ndviAnom.npy')[0]
	eviAnom2013to17 = np.load(wdvars+sName+'/eviAnom.npy')[0]
	ndwiAnom2013to17 = np.load(wdvars+sName+'/ndwiAnom.npy')[0]

	ndviAvgAllYears = np.zeros(shape=(8,12))
	eviAvgAllYears = np.zeros(shape=(8,12))
	ndwiAvgAllYears = np.zeros(shape=(8,12))

	ndviAnomAllYears = np.zeros(shape=(8,12))
	eviAnomAllYears = np.zeros(shape=(8,12))
	ndwiAnomAllYears = np.zeros(shape=(8,12))

	ndviAvgAllYears[:5] = ndviAvg2013to17
	ndviAvgAllYears[5:] = ndviAvgCurrent
	ndviAnomAllYears[:5] = ndviAnom2013to17
	ndviAnomAllYears[5:] = ndviAnomCurrent

	eviAvgAllYears[:5] = eviAvg2013to17
	eviAvgAllYears[5:] = eviAvgCurrent
	eviAnomAllYears[:5] = eviAnom2013to17
	eviAnomAllYears[5:] = eviAnomCurrent

	ndwiAvgAllYears[:5] = ndwiAvg2013to17
	ndwiAvgAllYears[5:] = ndwiAvgCurrent
	ndwiAnomAllYears[:5] = ndwiAnom2013to17
	ndwiAnomAllYears[5:] = ndwiAnomCurrent
	####################################################

	if not os.path.exists(wdfigs+sName+'/2020'):
		os.makedirs(wdfigs+sName+'/2020')

variables = ['ndviAvgAllYears','eviAvgAllYears','ndwiAvgAllYears','ndviAnomAllYears','eviAnomAllYears','ndwiAnomAllYears']
varNameShort = ['ndviAvg','eviAvg','ndwiAvg','ndviAnom','eviAnom','ndwiAnom']
varNames = ['NDVI Avg','EVI Avg','NDWI Avg','NDVI Anom','EVI Anom','NDWI Anom']
for ivar in range(len(variables)):
	ydata = np.ma.compressed(np.ma.masked_array(vars()[variables[ivar]],vars()[variables[ivar]]==0))
	plt.clf()
	plt.plot(ydata, '*-b')
	plt.grid(True)
	plt.ylabel(varNames[ivar])
	plt.title(sName + ' ' + varNames[ivar] + ', 2013-2020')
	plt.savefig(wdfigs+sName+'/2020/'+sName+'_'+varNameShort[ivar]+'_2013-2020.pdf')
	exit()

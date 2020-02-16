import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
import sys
from sys import exit
import sklearn
from sklearn import svm
import time from sklearn.preprocessing import StandardScaler

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

startMonth=4
endMonth=13
#endMonth=int(time.datetime.datetime.now().strftime("%Y-%m-%d").split('-')[1])
#nmonths=endMonth
nmonths=13
makePlots=False

monthName=['January','Febuary','March','April','May','June','July','August','September','October','November','December']

for icountry in range(47):
	#icountry=26
	if icountry==0:
		continue

	Good=False

	ndviAnom=np.zeros(shape=(nmonths))
	eviAnom=np.zeros(shape=(nmonths))
	ndwiAnom=np.zeros(shape=(nmonths))
	
	ndviAvg=np.zeros(shape=(nmonths))
	eviAvg=np.zeros(shape=(nmonths))
	ndwiAvg=np.zeros(shape=(nmonths))
	
	########### find countries with the right growing season ###########
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

			if corrMonth>=7: # July
				print '\nRunning',country, ' month = '+monthName[corrMonth-1]
				Good=True
				break
			else:
				print country, 'has other season'
				break
	if Good==False:
		continue
	####################################################################

	counterSum=np.zeros(shape=(nmonths))
	counterSumforAvg=np.zeros(shape=(nmonths))
	ndviAnomSum=np.zeros(shape=(nmonths))
	eviAnomSum=np.zeros(shape=(nmonths))
	ndwiAnomSum=np.zeros(shape=(nmonths))

	ndviAvgSum=np.zeros(shape=(nmonths))
	eviAvgSum=np.zeros(shape=(nmonths))
	ndwiAvgSum=np.zeros(shape=(nmonths))

	########### load 2020 vars ###########
	climoCounterAll = np.load(wdvars+sName+'/2020/climoCounterUnprocessed.npy')
	ndviMonthAvgU=np.load(wdvars+sName+'/2020/ndviMonthAvgUnprocessed.npy')
	eviMonthAvgU=np.load(wdvars+sName+'/2020/eviMonthAvgUnprocessed.npy')
	ndwiMonthAvgU=np.load(wdvars+sName+'/2020/ndwiMonthAvgUnprocessed.npy')
	npixels = climoCounterAll.shape[-1]

	climoCounterAll = np.reshape(climoCounterAll, (24,npixels,npixels), order='C')[:13]
	ndviMonthAvgU = np.reshape(ndviMonthAvgU, (24,npixels,npixels), order='C')[:13]
	eviMonthAvgU = np.reshape(eviMonthAvgU, (24,npixels,npixels), order='C')[:13]
	ndwiMonthAvgU = np.reshape(ndwiMonthAvgU, (24,npixels,npixels), order='C')[:13]
	
	# To get rid of extra years
	#climoCounterAll=climoCounterAll[0]
	#ndviMonthAvgU=ndviMonthAvgU[0]
	#eviMonthAvgU=eviMonthAvgU[0]
	#ndwiMonthAvgU=ndwiMonthAvgU[0]
	######################################

	########### Load Monthly Climatologies ###########
	ndviClimo = np.zeros(shape=(13,npixels,npixels))
	eviClimo = np.zeros(shape=(13,npixels,npixels))
	ndwiClimo = np.zeros(shape=(13,npixels,npixels))

	ndviClimo[0:12] = np.load(wdvars+sName+'/ndviClimo.npy')
	eviClimo[0:12] = np.load(wdvars+sName+'/eviClimo.npy')
	ndwiClimo[0:12] = np.load(wdvars+sName+'/ndwiClimo.npy')

	ndviClimo[12] = ndviClimo[0]
	eviClimo[12] = eviClimo[0]
	ndwiClimo[12] = ndwiClimo[0]
	##################################################

	vlen=climoCounterAll.shape[1]
	hlen=climoCounterAll.shape[2]

	ndviMonthAvg=np.zeros(shape=(ndviMonthAvgU.shape))
	eviMonthAvg=np.zeros(shape=(eviMonthAvgU.shape))
	ndwiMonthAvg=np.zeros(shape=(ndwiMonthAvgU.shape))

	ndviAnomAllPix=np.zeros(shape=(nmonths,vlen,hlen))
	eviAnomAllPix=np.zeros(shape=(nmonths,vlen,hlen))
	ndwiAnomAllPix=np.zeros(shape=(nmonths,vlen,hlen))
	
	########### Compute Pixel-wise Averages and Anomalies ###########
	for m in range(startMonth,endMonth):
		for v in range(vlen):
			for h in range(hlen):
				ndviMonthAvg[m,v,h]=ndviMonthAvgU[m,v,h]/climoCounterAll[m,v,h]
				eviMonthAvg[m,v,h]=eviMonthAvgU[m,v,h]/climoCounterAll[m,v,h]
				ndwiMonthAvg[m,v,h]=ndwiMonthAvgU[m,v,h]/climoCounterAll[m,v,h]
	
				ndviAnomAllPix[m,v,h]=ndviMonthAvg[m,v,h]-ndviClimo[m,v,h]
				eviAnomAllPix[m,v,h]=eviMonthAvg[m,v,h]-eviClimo[m,v,h]
				ndwiAnomAllPix[m,v,h]=ndwiMonthAvg[m,v,h]-ndwiClimo[m,v,h]
	#################################################################

	########### Compute Anomalies and Avgs for the whole tile ###########
	##### Find Sums #####
	for m in range(startMonth,endMonth):
		for v in range(vlen):
			for h in range(hlen):
				if math.isnan(ndviAnomAllPix[m,v,h])==False and ndviAnomAllPix[m,v,h]!=0.:
					counterSum[m]+=1
					ndviAnomSum[m]+=ndviAnomAllPix[m,v,h]
					eviAnomSum[m]+=eviAnomAllPix[m,v,h]
					ndwiAnomSum[m]+=ndwiAnomAllPix[m,v,h]

				if math.isnan(ndviMonthAvg[m,v,h])==False and ndviMonthAvg[m,v,h]!=0.:
					counterSumforAvg[m]+=1
					ndviAvgSum[m]+=ndviMonthAvg[m,v,h]
					eviAvgSum[m]+=eviMonthAvg[m,v,h]
					ndwiAvgSum[m]+=ndwiMonthAvg[m,v,h]

	##### Divide Sums #####
	ndviAnom = np.nan_to_num(ndviAnomSum/counterSum)
	eviAnom = np.nan_to_num(eviAnomSum/counterSum)
	ndwiAnom = np.nan_to_num(ndwiAnomSum/counterSum)
	
	ndviAvg = np.nan_to_num(ndviAvgSum/counterSumforAvg)
	eviAvg = np.nan_to_num(eviAvgSum/counterSumforAvg)
	ndwiAvg = np.nan_to_num(ndwiAvgSum/counterSumforAvg)
	#####################################################################
	
	print ndviAnom[:]
	print ndviAvg[:],'\n'
	
	np.save(wdvars+sName+'/2020/ndviAnom',ndviAnom)
	np.save(wdvars+sName+'/2020/eviAnom',eviAnom)
	np.save(wdvars+sName+'/2020/ndwiAnom',ndwiAnom)
		
	np.save(wdvars+sName+'/2020/ndviAvg',ndviAvg)
	np.save(wdvars+sName+'/2020/eviAvg',eviAvg)
	np.save(wdvars+sName+'/2020/ndwiAvg',ndwiAvg)

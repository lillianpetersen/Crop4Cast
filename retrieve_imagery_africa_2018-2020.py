import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
import sys
from sys import exit
#import sklearn
#from sklearn import svm
import time
#from sklearn.preprocessing import StandardScaler
import datetime
#from celery import Celery
from scipy import ndimage

####################
# Function		 #
####################
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
####################		

time1 = np.round(time.time(),2)

wd = '/Users/lilllianpetersen/Google Drive/science_fair/'
wdvars = '/Users/lilllianpetersen/science_fair_2018/saved_vars/'
wdfigs = '/Users/lilllianpetersen/science_fair_2018/figures/'
wddata = '/Users/lilllianpetersen/science_fair_2018/data/'

countylats = np.load(wdvars+'county_lats.npy')
countylons = np.load(wdvars+'county_lons.npy')
countyName = np.load(wdvars+'countyName.npy')
stateName = np.load(wdvars+'stateName.npy')

start = '2018-01-01'
now = datetime.datetime.now()
end = now.strftime("%Y-%m-%d")
nyears = now.year - 2018 +1
nmonths = 12
makePlots = False
print 'makePlots = ',makePlots
padding = 0
res = 120

### Get list of countries
fmonths=open(wddata+'max_ndviMonths_final.csv')
corrMonthArray=99*np.ones(shape=(48))
countryList = np.zeros(shape=48,dtype='object')
for line in fmonths:
	tmp1=line.split(',')
	icountry=int(tmp1[0])
	countryList[icountry] = tmp1[1]
	corrMonth=tmp1[2][:-1]
	if len(corrMonth)>2: # if two seasons
		months=corrMonth.split('/')
		month1=int(months[0])
		month2=int(months[1])
		#currentMonth=int(datetime.datetime.now().strftime("%Y-%m-%d")[5:7])
		#month1toNow=-1*month1-currentMonth
		#month2toNow=-1*month2-currentMonth
		#if month2toNow<0:
		corrMonth = month1
		#elif month2toNow>0:
		#	corrMonth=month2

	corrMonthArray[icountry]=corrMonth

countriesWithCurrentSeason=[]
for icountry in range(48):
	corrMonth = corrMonthArray[icountry]
	if (corrMonth<7 and corrMonth!=99): # Height of season August or later
		print icountry, corrMonth, countryList[icountry]
		countriesWithCurrentSeason.append(icountry)

### Loop through countries to get satellite imagery
#for countryNum in countriesWithCurrentSeason:
for countryNum in range(1,len(countryList)+1):
	if countryNum!=44: continue # start at Tanzania

	f = open(wddata+'africa_latlons.csv')
	for line in f:
		tmp = line.split(',')
		if tmp[0]==str(countryNum):
			country = tmp[1]
			startlat = float(tmp[2])
			startlon = float(tmp[3])
			pixels = int(tmp[4])
	
	dltile = dl.raster.dltile_from_latlon(startlat,startlon,res,pixels,padding)
	print '\n\n'
	print country
	print start,end
	
	### Retrieve list of images over area for selected time
	images =  dl.metadata.search(
		#products = 'modis:09:CREFL',
		products = 'modis:09:v2',
		start_time = start,
		end_time = end,
		cloud_fraction = .8,
		geom = dltile['geometry'],
		limit = 10000,
		)
	
	lat = dltile['geometry']['coordinates'][0][0][0]
	lon = dltile['geometry']['coordinates'][0][0][1]
	
	
	n_images = len(images['features'])
	print('Number of image matches: %d' % n_images)
	
	#band_info = dl.metadata.bands(products='modis:09:CREFL')
	band_info = dl.metadata.bands(products='modis:09:v2')
	
	sName = country
	cName = 'growing_region'
	
	####################
	# Define Variables #
	####################
	print pixels,'\n'
	ndviAnom = -9999*np.ones(shape = (nyears,nmonths,pixels,pixels))
	ndviMonthAvg = np.zeros(shape=(nyears,nmonths,pixels,pixels))
	eviMonthAvg = np.zeros(shape=(nyears,nmonths,pixels,pixels))
	ndwiMonthAvg = np.zeros(shape=(nyears,nmonths,pixels,pixels))
	ndviClimo = np.zeros(shape=(nmonths,pixels,pixels))
	eviClimo = np.zeros(shape=(nmonths,pixels,pixels))
	ndwiClimo = np.zeros(shape=(nmonths,pixels,pixels))
	climoCounter = np.zeros(shape=(nyears,nmonths,pixels,pixels))
	plotYear = np.zeros(shape=(nyears+1,nmonths,150))
	monthAll = np.zeros(shape=(n_images))
	ndviAll = -9999*np.ones(shape=(150,pixels,pixels))
	eviAll = -9999*np.ones(shape=(150,pixels,pixels))
	Mask = np.ones(shape=(150,pixels,pixels)) 
	ndwiAll = np.zeros(shape=(150,pixels,pixels))
	####################
	nGoodImage=-1
	d=-1

	for nImage in range(n_images):
	
		monthAll[nImage]=str(images['features'][nImage]['id'][17:19])
	
		if monthAll[nImage]!=monthAll[nImage-1] and nImage!=0:
			print sName
			d=-1
			for v in range(pixels):
				for h in range(pixels):
					for d in range(150):
						if Mask[d,v,h]==0:
							ndviMonthAvg[y,m,v,h]+=ndviAll[d,v,h]
							eviMonthAvg[y,m,v,h]+=eviAll[d,v,h]
							ndwiMonthAvg[y,m,v,h]+=ndwiAll[d,v,h]
							climoCounter[y,m,v,h]+=1 # number of good days in a month
					ndviClimo[m,v,h]+=ndviMonthAvg[y,m,v,h]
					eviClimo[m,v,h]+=eviMonthAvg[y,m,v,h]
					ndwiClimo[m,v,h]+=ndwiMonthAvg[y,m,v,h]
	
			if not os.path.exists(wdvars+sName+'/since2018'):
				os.makedirs(wdvars+sName+'/since2018')		 
			np.save(wdvars+sName+'/since2018/ndviClimoUnprocessed',ndviClimo)
			np.save(wdvars+sName+'/since2018/eviClimoUnprocessed',eviClimo)
			np.save(wdvars+sName+'/since2018/ndwiClimoUnprocessed',ndwiClimo)
			np.save(wdvars+sName+'/since2018/climoCounterUnprocessed',climoCounter)
			np.save(wdvars+sName+'/since2018/ndviMonthAvgUnprocessed',ndviMonthAvg)
			np.save(wdvars+sName+'/since2018/eviMonthAvgUnprocessed',eviMonthAvg) 
			np.save(wdvars+sName+'/since2018/ndwiMonthAvgUnprocessed',ndwiMonthAvg)
	
			d=-1
			ndviAll=-9999*np.ones(shape=(150,pixels,pixels))
			eviAll=-9999*np.ones(shape=(150,pixels,pixels))
			Mask=np.ones(shape=(150,pixels,pixels)) 
			ndwiAll=np.zeros(shape=(150,pixels,pixels))
	
	
		# get the scene id
		scene = images['features'][nImage]['id']
		###############################################
		# NDVI
		###############################################
		# load the image data into a numpy array
	
		band_info_index={}
		for i in range(len(band_info)):
			band_info_index[band_info[i]['name']]=i
	
		try:
			default_range = band_info[band_info_index['ndvi']]['default_range']
			physical_range = band_info[band_info_index['ndvi']]['physical_range']
			arrNDVI, meta = dl.raster.ndarray(
				scene,
				resolution=dltile['properties']['resolution'],
				bounds=dltile['properties']['outputBounds'],
				srs=dltile['properties']['cs_code'],
				bands=['ndvi', 'alpha'],
				scales=[[default_range[0], default_range[1], physical_range[0], physical_range[1]]],
				#scales=[[0,16383,-1.0,1.0]],
				data_type='Float32',
				)
		except:
			print('ndvi: %s could not be retreived' % scene)
			continue
		if np.amax(arrNDVI)==0: continue
	
		#######################
		# Get cloud data	  #
		#######################
		try:
			default_range = band_info[band_info_index['alpha']]['default_range']
			data_range = band_info[band_info_index['alpha']]['data_range']
			cloudMask, meta = dl.raster.ndarray(
				scene,
				resolution=dltile['properties']['resolution'],
				bounds=dltile['properties']['outputBounds'],
				srs=dltile['properties']['cs_code'],
				bands=['visual_cloud_mask', 'alpha'],
				#scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
				#scales=[[0, 65535, 0., 1.]],
				data_type='Float32',
				)
		except:
			print('cloudMask: %s could not be retreived' % scene)
			nGoodImage-=1
			d-=1
			continue 
	
		cloudMask=cloudMask[:,:,0]
		#######################
		
		###############################################
		# Test for bad days
		############################################### 
	
		maskforAlpha = arrNDVI[:, :, 1] == 0 
	
		if np.sum(cloudMask)>0.85*(pixels*pixels):
			print 'clouds: continued', np.round(np.sum(cloudMask)/(pixels*pixels),3)
			continue
		
		if np.amin(maskforAlpha)==1:
			print 'bad: continued'
			continue
	
		nGoodImage+=1
		d+=1
		
		###############################################
		# time
		############################################### 
	
		#xtime=str(images['features'][nImage]['id'][20:30]) # for CREFL
		xtime=str(images['features'][nImage]['id'][12:22]) # for v2
		date=xtime
		year=int(xtime[0:4])
		if nGoodImage==0:
			startyear=year
		y=int(year-startyear)
		month=int(xtime[5:7])
		m=int(month-1)
		day=xtime[8:10]
		dayOfYear=(float(month)-1)*30+float(day)
		plotYear[y,m,d]=year+dayOfYear/365.0
		
		###############################################
		# Back to NDVI
		############################################### 
	
		time2=time.time()
	
		print date, nGoodImage, np.round(np.sum(cloudMask)/(pixels*pixels),3), d, np.round(time2-time1,1)
		sys.stdout.flush()
	
		time1=time.time()
	
		Mask[d,:,:]=cloudMask+maskforAlpha
		Mask[Mask>1]=1
	
		if np.amin(Mask[d])==1:
			print 'bad: continued'
			nGoodImage-=1
			d-=1
			continue
	
		if makePlots:
			if not os.path.exists(wdfigs+sName+'/'+cName):
				os.makedirs(wdfigs+sName+'/'+cName)
			masked_ndvi = np.ma.masked_array(arrNDVI[:, :, 0], Mask[d,:,:])
			plt.figure(figsize=[10,10])
			plt.imshow(masked_ndvi, cmap=my_cmap, vmin=-.4, vmax=.9)
			plt.title('NDVI: '+cName+', '+sName+', '+str(date), fontsize=20)
			cb = plt.colorbar()
			cb.set_label("NDVI")
			plt.savefig(wdfigs+sName+'/'+cName+'/ndvi_'+str(date)+'_'+str(nGoodImage)+'.jpg')
			plt.clf() 
	
		ndviAll[d,:,:]=np.ma.masked_array(arrNDVI[:,:,0],Mask[d,:,:])
	
		###############################################
		# Cloud
		###############################################
	
		if makePlots:
			plt.clf()
			plt.figure(figsize=[10,10])
			plt.imshow(np.ma.masked_array(cloudMask,Mask[d,:,:]), cmap='gray', vmin=0, vmax=1)
			plt.title('Cloud Mask: '+cName+', '+sName+', '+str(date), fontsize=20)
			plt.savefig(wdfigs+sName+'/'+cName+'/cloud_'+str(date)+'_'+str(nGoodImage)+'.jpg')
			plt.clf()
			
	
		###############################################
		# EVI
		###############################################
		try:
			arrEVI, meta = dl.raster.ndarray(
				scene,
				resolution=dltile['properties']['resolution'],
				bounds=dltile['properties']['outputBounds'],
				srs=dltile['properties']['cs_code'],
				bands=['derived:evi', 'alpha'],
				scales=[[0,65535,-1.,1.]], # from Rick
				#scales=[[0,2**16,-1.,1.]],
				#scales=[[0,16383,-1.0,1.0]],
				data_type='Float32',
				)
		except:
			print('evi: %s could not be retreived' % scene)
			nGoodImage-=1
			d-=1
			continue
	
		if makePlots:
			plt.clf() 
			masked_evi = np.ma.masked_array(arrEVI[:, :, 0], Mask[d,:,:])
			plt.figure(figsize=[10,10])
			plt.imshow(masked_evi, cmap=my_cmap, vmin=-.4, vmax=.9)
			#plt.imshow(masked_ndvi, cmap='jet')#, vmin=0, vmax=65535)
			plt.title('EVI: '+cName+', '+sName+', '+str(date), fontsize=20)
			cb = plt.colorbar()
			cb.set_label("EVI")
			plt.savefig(wdfigs+sName+'/'+cName+'/evi_'+str(date)+'_'+str(nGoodImage)+'.jpg')
			plt.clf() 
	
		eviAll[d,:,:]=np.ma.masked_array(arrEVI[:,:,0],Mask[d,:,:])
	
		###############################################
		# NDWI
		###############################################
		
		try:
			default_range = band_info[band_info_index['nir']]['default_range']
			data_range = band_info[band_info_index['nir']]['physical_range']
			nir, meta = dl.raster.ndarray(
				scene,
				resolution=dltile['properties']['resolution'],
				bounds=dltile['properties']['outputBounds'],
				srs=dltile['properties']['cs_code'],
				bands=['nir', 'alpha'],
				scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
				#scales=[[0,2**14,-1.,1.]],
				data_type='Float32',
				)
		except:
			print('nir: %s could not be retreived' % scene)
			nGoodImage-=1
			d-=1
			continue
	
		nirM=np.ma.masked_array(nir[:,:,0],Mask[d,:,:])
		
		try:
			default_range = band_info[band_info_index['green']]['default_range']
			data_range = band_info[band_info_index['green']]['physical_range']
			green, meta = dl.raster.ndarray(
				scene,
				resolution=dltile['properties']['resolution'],
				bounds=dltile['properties']['outputBounds'],
				srs=dltile['properties']['cs_code'],
				bands=['green', 'alpha'],
				scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
				#scales=[[0,2**14,-1.,1.]],
				data_type='Float32',
				)
		except:
			print('green: %s could not be retreived' % scene)
			nGoodImage-=1
			d-=1
			continue
		  
		greenM=np.ma.masked_array(green[:,:,0],Mask[d,:,:])
	
		ndwiAll[d,:,:] = (green[:,:,0]-nir[:,:,0])/(nir[:,:,0]+green[:,:,0]+1e-9)
	
		if makePlots:
			masked_ndwi = np.ma.masked_array(ndwiAll[d,:,:], Mask[d,:,:])
			plt.figure(figsize=[10,10])
			plt.imshow(masked_ndwi, cmap='jet', vmin=-1, vmax=1)
			plt.title('NDWI:' +cName+', '+sName+', '+str(date), fontsize=20)
			cb = plt.colorbar()
			cb.set_label("NDWI")
			plt.savefig(wdfigs+sName+'/'+cName+'/ndwi_'+str(date)+'_'+str(nGoodImage)+'.jpg')
			plt.clf()
		
		###############################################
		# Visual
		###############################################
		ids = [f['id'] for f in images['features']]
	
		if makePlots:
			arr, meta = dl.raster.ndarray(
				scene,
				resolution=dltile['properties']['resolution'],
				bounds=dltile['properties']['outputBounds'],
				srs=dltile['properties']['cs_code'],
				bands=['red', 'green', 'blue', 'alpha'],
				scales=[[0,4000], [0, 4000], [0, 4000], None],
				data_type='Byte',
				)
	
			plt.figure(figsize=[10,10])
			plt.imshow(arr)
			plt.title('Visible: '+cName+', '+sName+', '+str(date), fontsize=20)
			plt.savefig(wdfigs+sName+'/'+cName+'/visual_'+str(date)+'_'+str(nGoodImage)+'.jpg')
	
		if makePlots:
			if nGoodImage==5:
				exit()
	
		
	########################
	# Save variables	   #
	######################## 
	
	for v in range(pixels):
		for h in range(pixels):
			for d in range(150):
				if Mask[d,v,h]==0:
					ndviMonthAvg[y,m,v,h] += ndviAll[d,v,h]
					eviMonthAvg[y,m,v,h] += eviAll[d,v,h]
					ndwiMonthAvg[y,m,v,h] += ndwiAll[d,v,h]
					climoCounter[y,m,v,h] += 1 # number of good days in a month
			ndviClimo[m,v,h] += ndviMonthAvg[y,m,v,h]
			eviClimo[m,v,h] += eviMonthAvg[y,m,v,h]
			ndwiClimo[m,v,h] += ndwiMonthAvg[y,m,v,h]
	
	np.save(wdvars+sName+'/since2018/ndviClimoUnprocessed',ndviClimo)
	np.save(wdvars+sName+'/since2018/eviClimoUnprocessed',eviClimo)
	np.save(wdvars+sName+'/since2018/ndwiClimoUnprocessed',ndwiClimo)
	np.save(wdvars+sName+'/since2018/climoCounterUnprocessed',climoCounter)
	np.save(wdvars+sName+'/since2018/ndviMonthAvgUnprocessed',ndviMonthAvg)
	np.save(wdvars+sName+'/since2018/eviMonthAvgUnprocessed',eviMonthAvg) 
	np.save(wdvars+sName+'/since2018/ndwiMonthAvgUnprocessed',ndwiMonthAvg)







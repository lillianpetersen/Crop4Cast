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
### Running mean/Moving average
def mask_water(image):
	shape = image.shape
	length = image.size

	# reshape to linear
	x = image.reshape(length)

	# slice every 4th element
	y = x[0::4]

	# mask if less than 60 for NIR
	sixty = np.ones(len(y))*60
	z = y < sixty

	# multiply by 4
	oceanMask = np.repeat(z, 4)

	# apply mask to original array
	masked = np.ma.masked_array(x, oceanMask)
	b = np.ma.filled(masked, 0)

	# reshape
	c = b.reshape(shape)
	masked = masked.reshape(shape)
	oceanMask = oceanMask.reshape(shape)
	oceanMask = oceanMask[:,:,0]
	return c, oceanMask

def ltk_cloud_mask(X, get_rgb=False):
	#
	#   Modified Luo et al. (2008) LTK scheme (Oreopoulos et al. 2011)
	#   https://landsat.usgs.gov/documents/Oreopoulos_cloud.jpg
	#
	#	inputs:
	#	X	   6-band landsat images : VIS/NIR/SWIR bands[1,2,3,4,5,7] in top-of-atmosphere reflectance
	#
	#	output:
	#	Y	   byte-valued cloud/snow/water/shadow mask
	#	vals:   (based on official NASA LTK cloud mask labels)
	#	1	   land
	#	2	   snow
	#	3	   water bodies
	#	4	   clouds
	#	5	   vegetation
	#

	L1 = X[:,:,0]
	L3 = X[:,:,1]
	L4 = X[:,:,2]
	L5 = X[:,:,3]

	Y = np.zeros(L1.shape, dtype='uint8')

	# stage 1 : non-vegetated land
	#
	indexA = (L1 < L3)
	indexA = np.logical_and(indexA, (L3 < L4))
	indexA = np.logical_and(indexA, (L4 < np.multiply(L5, 1.07)))
	indexA = np.logical_and(indexA, (L5 < 0.65))

	indexB = (np.multiply(L1, 0.8) < L3)
	indexB = np.logical_and(indexB, (L3 < np.multiply(L4, 0.8)))
	indexB = np.logical_and(indexB, (L4 < L5))
	indexB = np.logical_and(indexB, (L3 < 0.22))

	index = np.logical_and((Y == 0), np.logical_or(indexA, indexB))
	Y[index] = 1  # non-vegetated lands

	# stage 2 : snow/ice
	#
	indexA = (L3 >= 0.24)
	indexA = np.logical_and(indexA, (L5 < 0.16))
	indexA = np.logical_and(indexA, (L3 > L4))

	indexB = (L3 > 0.18)
	indexB = np.logical_and(indexB, (L3 < 0.24))
	indexB = np.logical_and(indexB, (L5 < np.subtract(L3, 0.08)))
	indexB = np.logical_and(indexB, (L3 > L4))

	index = np.logical_and((Y == 0), np.logical_or(indexA, indexB))
	Y[index] = 2  # snow/ice

	# stage 3 : water bodies
	#
	indexA = (L3 > L4)
	indexA = np.logical_and(indexA, (L3 > np.multiply(L5, 0.67)))
	indexA = np.logical_and(indexA, (L1 < 0.30))
	indexA = np.logical_and(indexA, (L3 < 0.20))

	indexB = (L3 > np.multiply(L4, 0.8))
	indexA = np.logical_and(indexA, (L3 > np.multiply(L5, 0.67)))
	indexB = np.logical_and(indexB, (L3 < 0.06))

	index = np.logical_and((Y == 0), np.logical_or(indexA, indexB))
	Y[index] = 3  # water bodies

	# stage 4 : clouds
	#
	index = np.logical_or((L1 > 0.25), (L3 > 0.27))
	index = np.logical_and(index, (L5 > 0.22))
	index = np.logical_and(index, (np.maximum(L1, L3) > np.multiply(L5, 0.87)))

	index = np.logical_and((Y == 0), index)
	Y[index] = 4  # clouds

	# stage 5 : vegetation
	#
	Y[(Y == 0)] = 5  # vegetation

	#
	if get_rgb:
		rgb = rgb_clouds(Y)
		return Y, rgb
	#
	globals().update(())
	return Y

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

time1=np.round(time.time(),2)

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wdvars='/Users/lilllianpetersen/science_fair_2018/saved_vars/'
wdfigs='/Users/lilllianpetersen/science_fair_2018/figures/'
wddata='/Users/lilllianpetersen/science_fair_2018/data/'

countylats=np.load(wdvars+'county_lats.npy')
countylons=np.load(wdvars+'county_lons.npy')
countyName=np.load(wdvars+'countyName.npy')
stateName=np.load(wdvars+'stateName.npy')

recentSeason = False
since2018 = True

start='2013-01-01'
end='2020-02-00'
nyears=5
nmonths=12
makePlots=False
print 'makePlots=',makePlots
padding = 0
res=120

now = datetime.datetime.now()
#start='2019-01-01'
#end=now.strftime("%Y-%m-%d")
nyears=2
nmonths=12

fmonths=open(wddata+'max_ndviMonths_final.csv')
corrMonthArray=99*np.ones(shape=(48))
countryList=[]
for line in fmonths:
	tmp1=line.split(',')
	icountry=int(tmp1[0])
	countryList.append(tmp1[1])
	corrMonth=tmp1[2][:-1]
	if len(corrMonth)>2:
		months=corrMonth.split('/')
		month1=int(months[0])
		month2=int(months[1])
		currentMonth=int(datetime.datetime.now().strftime("%Y-%m-%d")[5:7])
		month1toNow=-1*month1-currentMonth
		month2toNow=-1*month2-currentMonth
		if month2toNow<0:
			corrMonth=month1
		elif month2toNow>0:
			corrMonth=month2
	corrMonthArray[icountry]=corrMonth
	#if tmp1[0]==countryNum:
	#	country=tmp1[1]
	#	corrMonth=tmp1[2][:-1]
	#	if len(corrMonth)>2:
	#		months=corrMonth.split('/')
	#		month1=corrMonth[0]
	#		month2=corrMonth[1]
	#		corrMonth=month1

countriesWithCurrentSeason=[]
for icountry in range(47):
	corrMonth=corrMonthArray[icountry]
	if (corrMonth>=6 and corrMonth!=99): # Height of season August or later
		#print icountry, corrMonth
		countriesWithCurrentSeason.append(icountry)

for countryNum in countriesWithCurrentSeason:
	if countryNum!=6: continue

	corrMonth = corrMonthArray[countryNum]+1 # 1=Jan, 2=Feb, etc
	
	if recentSeason:
		start='2019-0'+str(int(corrMonth-2))+'-01'
		if corrMonth<=7: # July
			end='2019-0'+str(int(corrMonth+2))+'-30'
		elif corrMonth==8 or corrMonth==9 or corrMonth==10: # August + September + October
			end='2019-'+str(int(corrMonth+2))+'-30'
		elif corrMonth==11: # November
			end='2020-01-30'
		elif corrMonth==12: # December
			start='2019-'+str(int(corrMonth-2))+'-01'
			end='2020-02-30'
	elif since2018:
		start = '2018-01-10'
		end = '2019-12-30'

	f=open(wddata+'africa_latlons.csv')
	for line in f:
		tmp=line.split(',')
		if tmp[0]==str(countryNum):
			country=tmp[1]
			startlat=float(tmp[2])
			startlon=float(tmp[3])
			pixels=int(tmp[4])

	dltile=dl.raster.dltile_from_latlon(startlat,startlon,res,pixels,padding)
	print '\n\n'
	print country
	print start,end
	
	images= dl.metadata.search(
		#products='5151d2825f5e29ff129f86d834946363ff3f7e57:modis:09:CREFL_v2_test',
		products='modis:09:CREFL',
		start_time=start,
		end_time=end,
		cloud_fraction=.8,
		geom=dltile['geometry'],
		limit=10000,
		)
	
	lat=dltile['geometry']['coordinates'][0][0][0]
	lon=dltile['geometry']['coordinates'][0][0][1]
	
	
	n_images = len(images['features'])
	print('Number of image matches: %d' % n_images)
	
	#band_info=dl.metadata.bands(products='landsat:LT05:PRE:TOAR')
	#band_info=dl.metadata.bands(products='5151d2825f5e29ff129f86d834946363ff3f7e57:modis:09:CREFL_v2_test')
	band_info=dl.metadata.bands(products='modis:09:CREFL')
	
	sName=country
	cName='growing_region'
	
	####################
	# Define Variables #
	####################
	print pixels,'\n'
	ndviAnom=-9999*np.ones(shape=(nyears,nmonths,pixels,pixels))
	ndviMonthAvg=np.zeros(shape=(nyears,nmonths,pixels,pixels))
	eviMonthAvg=np.zeros(shape=(nyears,nmonths,pixels,pixels))
	ndwiMonthAvg=np.zeros(shape=(nyears,nmonths,pixels,pixels))
	ndviClimo=np.zeros(shape=(nmonths,pixels,pixels))
	eviClimo=np.zeros(shape=(nmonths,pixels,pixels))
	ndwiClimo=np.zeros(shape=(nmonths,pixels,pixels))
	climoCounter=np.zeros(shape=(nyears,nmonths,pixels,pixels))
	plotYear=np.zeros(shape=(nyears+1,nmonths,140))
	monthAll=np.zeros(shape=(n_images))
	ndviAll=-9999*np.ones(shape=(140,pixels,pixels))
	eviAll=-9999*np.ones(shape=(140,pixels,pixels))
	Mask=np.ones(shape=(140,pixels,pixels)) 
	ndwiAll=np.zeros(shape=(140,pixels,pixels))
	####################
	nGoodImage=-1
	d=-1
	for nImage in range(n_images):
	
		monthAll[nImage]=str(images['features'][nImage]['id'][25:27])
	
		if monthAll[nImage]!=monthAll[nImage-1] and nImage!=0:
			d=-1
			for v in range(pixels):
				for h in range(pixels):
					for d in range(140):
						if Mask[d,v,h]==0:
							ndviMonthAvg[y,m,v,h]+=ndviAll[d,v,h]
							eviMonthAvg[y,m,v,h]+=eviAll[d,v,h]
							ndwiMonthAvg[y,m,v,h]+=ndwiAll[d,v,h]
							climoCounter[y,m,v,h]+=1 # number of good days in a month
					ndviClimo[m,v,h]+=ndviMonthAvg[y,m,v,h]
					eviClimo[m,v,h]+=eviMonthAvg[y,m,v,h]
					ndwiClimo[m,v,h]+=ndwiMonthAvg[y,m,v,h]
	
			if recentSeason:
				if not os.path.exists(wdvars+country+'/2020'):
					os.makedirs(wdvars+country+'/2020')		 
				np.save(wdvars+country+'/2020/ndviClimoUnprocessed',ndviClimo)
				np.save(wdvars+country+'/2020/eviClimoUnprocessed',eviClimo)
				np.save(wdvars+country+'/2020/ndwiClimoUnprocessed',ndwiClimo)
				np.save(wdvars+country+'/2020/climoCounterUnprocessed',climoCounter)
				np.save(wdvars+country+'/2020/ndviMonthAvgUnprocessed',ndviMonthAvg)
				np.save(wdvars+country+'/2020/eviMonthAvgUnprocessed',eviMonthAvg) 
				np.save(wdvars+country+'/2020/ndwiMonthAvgUnprocessed',ndwiMonthAvg)
			elif since2018:
				if not os.path.exists(wdvars+country+'/since2018'):
					os.makedirs(wdvars+country+'/since2018')		 
				np.save(wdvars+country+'/since2018/ndviClimoUnprocessed',ndviClimo)
				np.save(wdvars+country+'/since2018/eviClimoUnprocessed',eviClimo)
				np.save(wdvars+country+'/since2018/ndwiClimoUnprocessed',ndwiClimo)
				np.save(wdvars+country+'/since2018/climoCounterUnprocessed',climoCounter)
				np.save(wdvars+country+'/since2018/ndviMonthAvgUnprocessed',ndviMonthAvg)
				np.save(wdvars+country+'/since2018/eviMonthAvgUnprocessed',eviMonthAvg) 
				np.save(wdvars+country+'/since2018/ndwiMonthAvgUnprocessed',ndwiMonthAvg)
	
			d=-1
			ndviAll=-9999*np.ones(shape=(140,pixels,pixels))
			eviAll=-9999*np.ones(shape=(140,pixels,pixels))
			Mask=np.ones(shape=(140,pixels,pixels)) 
			ndwiAll=np.zeros(shape=(140,pixels,pixels))
	
	
		# get the scene id
		#scene = images['features'][indexSorted[nImage]]['key']
		#scene = images['features'][nImage]['key']
		scene = images['features'][nImage]['id']
		#print scene
		###############################################
		# NDVI
		###############################################
		# load the image data into a numpy array
	
		band_info_index={}
		for i in range(len(band_info)):
			band_info_index[band_info[i]['name']]=i
	
		try:
			default_range= band_info[band_info_index['ndvi']]['default_range']
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
	
		#cloudMask=1-cloudMask[:,:,0]
		cloudMask=cloudMask[:,:,0]
		#######################
		
		###############################################
		# Test for bad days
		############################################### 
	
		#take out days without data 
		#if arrCloud.shape == ()==True:
		#	continue
		#maskforCloud = arrCloud[:, :, 1] != 0 # False=Good, True=Bad
		#if np.sum(cloudMask)==0:
		#	print 'continued'
		#	continue
		#
		#swap = {5:0,4:1,1:0,2:0,3:0,0:1}
		#for v in range(pixels):
		#	for h in range(pixels):
		#if cloudMask[v,h]==3 and v<600:
		#	cloudMask[v,h]=1
		#else:
		#		cloudMask[v,h]=swap[cloudMask[v,h]]
	
		# take out days with too many clouds
		#cloudMask = arrCloud[:, :, 0] == 0 # True=good False=bad
		#if np.sum(cloudMask)>0.85*(np.sum(countyMaskOpposite)):
		#	print 'clouds: continued', np.round(np.sum(cloudMask)/(np.sum(countyMaskOpposite)),3)
		#	continue		
	
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
	
		#xtime=str(images['features'][nImage]['id'][64:74]) # MODIS test
		xtime=str(images['features'][nImage]['id'][20:30]) # old MODIS
		#xtime.append(str(images['features'][nImage]['properties']['acquired'][0:10])) Landsat
		date=xtime
		year=int(xtime[0:4])
		if nGoodImage==0:
			startyear=year
			#nyears=2018-startyear
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
		#cloudMask = arrCloud[:, :, 0] != 0 
		#cloudMask = arrCloud[:, :, 1] == 0 #for desert
	
		time1=time.time()
	
		Mask[d,:,:]=cloudMask+maskforAlpha
		Mask[Mask>1]=1
	
		if np.amin(Mask[d])==1:
			print 'bad: continued'
			nGoodImage-=1
			d-=1
			continue
	
		#if nGoodImage==1:
		#	exit()
		#Mask[:,:,nGoodImage]=1-Mask[:,:,nGoodImage]
		#Mask[:,:,nGoodImage]=ndimage.binary_dilation(Mask[:,:,nGoodImage],iterations=3)
		#Mask[:,:,nGoodImage]=1-Mask[:,:,nGoodImage]
		#cloudMask[:,:,nGoodImage]=1-cloudMask[:,:,nGoodImage]
		#cloudMask[:,:,nGoodImage]=ndimage.binary_dilation(Mask[:,:,nGoodImage],iterations=3)
		#cloudMask[:,:,nGoodImage]=1-cloudMask[:,:,nGoodImage]
	
		if makePlots:
			if not os.path.exists(wdfigs+sName+'/'+cName):
				os.makedirs(wdfigs+sName+'/'+cName)
			masked_ndvi = np.ma.masked_array(arrNDVI[:, :, 0], Mask[d,:,:])
			plt.figure(figsize=[10,10])
			plt.imshow(masked_ndvi, cmap=my_cmap, vmin=-.4, vmax=.9)
			#plt.imshow(masked_ndvi, cmap='jet')#, vmin=0, vmax=65535)
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
			#cb = plt.colorbar()
			#cb.set_label("Cloud")
			plt.savefig(wdfigs+sName+'/'+cName+'/cloud_'+str(date)+'_'+str(nGoodImage)+'.jpg')
			plt.clf()
			
	
		###############################################
		# EVI
		###############################################
		try:
			#default_range= band_info[band_info_index['evi']]['default_range']
			#physical_range = band_info[band_info_index['evi']]['physical_range']
			arrEVI, meta = dl.raster.ndarray(
				scene,
				resolution=dltile['properties']['resolution'],
				bounds=dltile['properties']['outputBounds'],
				srs=dltile['properties']['cs_code'],
				bands=['evi', 'alpha'],
				#scales=[[default_range[0], default_range[1], physical_range[0], physical_range[1]]],
				scales=[[0,2**16,-1.,1.]],
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
			for d in range(140):
				if Mask[d,v,h]==0:
					ndviMonthAvg[y,m,v,h]+=ndviAll[d,v,h]
					eviMonthAvg[y,m,v,h]+=eviAll[d,v,h]
					ndwiMonthAvg[y,m,v,h]+=ndwiAll[d,v,h]
					climoCounter[y,m,v,h]+=1 # number of good days in a month
			ndviClimo[m,v,h]+=ndviMonthAvg[y,m,v,h]
			eviClimo[m,v,h]+=eviMonthAvg[y,m,v,h]
			ndwiClimo[m,v,h]+=ndwiMonthAvg[y,m,v,h]
	
	if recentSeason:
		np.save(wdvars+country+'/2020/ndviClimoUnprocessed',ndviClimo)
		np.save(wdvars+country+'/2020/eviClimoUnprocessed',eviClimo)
		np.save(wdvars+country+'/2020/ndwiClimoUnprocessed',ndwiClimo)
		np.save(wdvars+country+'/2020/climoCounterUnprocessed',climoCounter)
		np.save(wdvars+country+'/2020/ndviMonthAvgUnprocessed',ndviMonthAvg)
		np.save(wdvars+country+'/2020/eviMonthAvgUnprocessed',eviMonthAvg) 
		np.save(wdvars+country+'/2020/ndwiMonthAvgUnprocessed',ndwiMonthAvg)
	if since2018:
		np.save(wdvars+country+'/since2018/ndviClimoUnprocessed',ndviClimo)
		np.save(wdvars+country+'/since2018/eviClimoUnprocessed',eviClimo)
		np.save(wdvars+country+'/since2018/ndwiClimoUnprocessed',ndwiClimo)
		np.save(wdvars+country+'/since2018/climoCounterUnprocessed',climoCounter)
		np.save(wdvars+country+'/since2018/ndviMonthAvgUnprocessed',ndviMonthAvg)
		np.save(wdvars+country+'/since2018/eviMonthAvgUnprocessed',eviMonthAvg) 
		np.save(wdvars+country+'/since2018/ndwiMonthAvgUnprocessed',ndwiMonthAvg)







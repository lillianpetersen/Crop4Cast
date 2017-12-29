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
from celery import Celery
from scipy import ndimage

####################
# Function		 #
####################
### Running mean/Moving average
def movingaverage(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')
	
def variance(x):   
	'''function to compute the variance (std dev squared)'''
	xAvg=np.mean(x)
	xOut=0.
	for k in range(len(x)):
		xOut=xOut+(x[k]-xAvg)**2
	xOut=xOut/(k+1)
	return xOut

def rolling_median(var,window):
	'''var: array-like. One dimension
	window: Must be odd'''
	n=len(var)
	halfW=int(window/2)
	med=np.zeros(shape=(var.shape))
	for j in range(halfW,n-halfW):
		med[j]=np.ma.median(var[j-halfW:j+halfW+1])
	 
	for j in range(0,halfW):
		w=2*j+1
		med[j]=np.ma.median(var[j-w/2:j+w/2+1])
		i=n-j-1
		med[i]=np.ma.median(var[i-w/2:i+w/2+1])
	
	return med	
	
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
	#   https://landsat.usgs.gov/documents/Oreopoulos_cloud.pdf
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

#celery = Celery('compute_ndvi_forCloud', broker='redis://localhost:6379/0')
#
##wd='gs://lillian-bucket-storage/'
wd='/Users/lilllianpetersen/Google Drive/science_fair/'

countylats=np.load(wd+'saved_vars/county_lats.npy')
countylons=np.load(wd+'saved_vars/county_lons.npy')
countyName=np.load(wd+'saved_vars/countyName.npy')
stateName=np.load(wd+'saved_vars/stateName.npy')

# Celery task goes into start-up script

#vlen=256
#hlen=256
start='2001-01-01'
#end='2016-12-31'
end='2001-12-31'
#nyears=17
nyears=1
country='US'
##country='Ethiopia'
makePlots=False
print 'makePlots=',makePlots
#padding = 0
#pixels = vlen+2*padding
#res = 30
res=120

#vlen=100
#hlen=100
#padding=0
#pixels=vlen+2*padding
#	
badarrays=0
for icounty in range(len(countylats)):

	clat=countylats[icounty]
	clon=countylons[icounty]
	cName=countyName[icounty].title()
	sName=stateName[icounty].title()

	cNamel=cName.lower()
	sNamel=sName.lower()


	if sName!='Illinois':
		continue
	if clat<38:
		continue
	#print sName,cName,clat,clon

	#matches=dl.places.find('united-states_washington')
	#matches=dl.places.find('north-america_united-states')
	#matches=dl.places.find('united-states_iowa')
	#matches=dl.places.find('puerto-rico_san-juan')
	#matches=dl.places.find('africa_ethiopia')
	
	matches=dl.places.find('united-states_'+sNamel+'_'+cNamel)
	aoi = matches[0]
	shape = dl.places.shape(aoi['slug'], geom='low')
	
	images= dl.metadata.search(
		products='modis:09:CREFL',
		start_time=start,
		end_time=end,
		cloud_fraction=.8,
		limit=10000,
		place=aoi['slug']
		)

	#dltiles = dl.raster.dltiles_from_shape(res, vlen, padding, shape)
	#dltile=dl.raster.dltile_from_latlon(clat,clon,res,vlen,padding)
	#dltile=dl.raster.dltile_from_latlon(7.5,37.5,res,vlen,padding)
	#dltile=dl.raster.dltile_from_latlon(7.902495, 38.034848,res,vlen,padding)
	#lonlist=np.zeros(shape=(len(dltiles['features'])))
	#latlist=np.zeros(shape=(len(dltiles['features'])))
	#for i in range(len(dltiles['features'])):
	#	lonlist[i]=dltiles['features'][i]['geometry']['coordinates'][0][0][0]
	#	latlist[i]=dltiles['features'][i]['geometry']['coordinates'][0][0][1]
	
	#features=np.zeros(shape=(len(dltiles),nyears,pixels*pixels,6))
	#target=np.zeros(shape=(len(dltiles),nyears,pixels*pixels))
	#features=np.zeros(shape=(len(dltile),nyears,pixels*pixels,6))
	#target=np.zeros(shape=(len(dltile),nyears,pixels*pixels))
	
	#@celery.task  
	#def tile_function(dltile,makePlots=False):
	
	#clas=["" for x in range(7)]
	#clasLong=["" for x in range(255)]
	#clasDict={}
	#clasNumDict={}
	#f=open(wd+'data/ground_data.txt')							    
	#for line in f:
	#	tmp=line.split(',')
	#	clasNumLong=int(tmp[0])
	#	clasLong[clasNumLong]=tmp[1]
	#	clasNum=int(tmp[3])
	#	clas[clasNum]=tmp[2]
	#	
	#	clasDict[clasLong[clasNumLong]]=clas[clasNum]
	#	clasNumDict[clasNumLong]=clasNum
	
	#lon=dltile['geometry']['coordinates'][0][0][0]
	#lat=dltile['geometry']['coordinates'][0][0][1]
	#print lon
	#print lat
	#latsave=str(lat)
	#latsave=latsave.replace('.','-')
	#lonsave=str(lon)
	#lonsave=lonsave.replace('.','-')
	
	print '\n\n'
	print cName,',',sName
	#print 'dltile: '+str(tile)+' of '+str(len(dltiles['features']))
	
	
	#oceanMask=np.zeros(shape=(pixels,pixels))
	
	#  for dltiles  #
	#images = dl.metadata.search(
	##products='landsat:LT05:PRE:TOAR',
	#	products='modis:09:CREFL',
	#	start_time=start,  #start='2000-01-01'
	#	end_time=end,   #end='2016-12-31'
	#	geom=dltile['geometry'],
	#	#cloud_fraction=0.8,
	#	limit = 10000
	#	)
	
	n_images = len(images['features'])
	print('Number of image matches: %d' % n_images)
	#avail_bands = dl.raster.get_bands_by_constellation("L5").keys()
	avail_bands = dl.raster.get_bands_by_constellation("MO").keys()
	print avail_bands 
	
	#band_info=dl.metadata.bands(products='landsat:LT05:PRE:TOAR')
	band_info=dl.metadata.bands(products='modis:09:CREFL')
	
	#dayOfYear=np.zeros(shape=(nyears,12))
	#year=np.zeros(shape=(n_images),dtype=int)
	#month=np.zeros(shape=(n_images),dtype=int)
	#day=np.zeros(shape=(n_images),dtype=int)
	#plotYear=np.zeros(shape=(nyears,12))
	#xtime=[]
	#i=-1
	#for feature in images['features']:
	#	i+=1
	#	# get the scene id
	#	scene = feature['id']
	#		
	#	xtime.append(str(images['features'][i]['id'][20:30]))
	#	#xtime.append(str(images['features'][i]['properties']['acquired'][0:10]))
	#	date=xtime[i]
	#	year[i]=xtime[i][0:4]
	#	if i==0:
	#		startyear=year[i]
	#		nyears=2018-startyear
	#	y=year[i]-startyear
	#	month[i]=xtime[i][5:7]
	#	m=month[i]-1
	#	day[i]=xtime[i][8:10]
	#	dayOfYear[y,m]=(float(month[i])-1)*30+float(day[i])
	#	plotYear[y,m]=year[i]+dayOfYear[y,m]/365.0
	#	
	#	
	#indexSorted=np.argsort(plotYear)

	scene = images['features'][0]['id']
	try:
		arrNDVI1, meta = dl.raster.ndarray(
			scene,
			resolution=res,
			bands=['ndvi', 'alpha'],
			cutline=shape['geometry']
			)
	except:
		print('ndvi: %s could not be retreived' % scene)
		continue
	
	vlen=arrNDVI1.shape[0]
	hlen=arrNDVI1.shape[1]
	
	print arrNDVI1.shape, scene

	scene = images['features'][1]['id']
	try:
		arrNDVI2, meta = dl.raster.ndarray(
			scene,
			resolution=res,
			bands=['ndvi', 'alpha'],
			cutline=shape['geometry']
			)
	except:
		print('ndvi: %s could not be retreived' % scene)
		continue

	print arrNDVI2.shape, scene

	scene = images['features'][2]['id']
	try:
		arrNDVI3, meta = dl.raster.ndarray(
			scene,
			resolution=res,
			bands=['ndvi', 'alpha'],
			cutline=shape['geometry']
			)
	except:
		print('ndvi: %s could not be retreived' % scene)
		continue

	print arrNDVI3.shape,scene

	if arrNDVI1.shape!=arrNDVI2.shape or arrNDVI2.shape!=arrNDVI3.shape or arrNDVI1.shape!=arrNDVI3.shape:
		#print 'ARRAYS DONT EQUAL EACHOTHER!!!'
		badarrays+=1
		continue

	countyMask=np.zeros(shape=(vlen,hlen))
	for v in range(vlen):
		for h in range(hlen):
			if arrNDVI1[v,h,0]==0 and arrNDVI2[v,h,0]==0 and arrNDVI3[v,h,0]==0:
				countyMask[v,h]=1

	####################
	# Define Variables #
	####################
	print vlen,hlen
	ndviAnom=-9999*np.ones(shape=(nyears,12,vlen,hlen))
	ndviMonthAvg=np.zeros(shape=(nyears,12,vlen,hlen))
	eviMonthAvg=np.zeros(shape=(nyears,12,vlen,hlen))
	ndwiMonthAvg=np.zeros(shape=(nyears,12,vlen,hlen))
	ndviClimo=np.zeros(shape=(12,vlen,hlen))
	eviClimo=np.zeros(shape=(12,vlen,hlen))
	ndwiClimo=np.zeros(shape=(12,vlen,hlen))
	climoCounter=np.zeros(shape=(nyears,12,vlen,hlen))
	plotYear=np.zeros(shape=(nyears,12,45))
	#year=np.zeros(shape=(n_images))
	#month=np.zeros(shape=(n_images))
	monthAll=np.zeros(shape=(n_images))
	#ndviHist=np.zeros(shape=(45,nyears,12))
	#ndviAvg=np.zeros(shape=(nyears,12))
	#ndviMed=np.zeros(shape=(nyears,12))
	xtime=[]
	
	#ndviHist=np.zeros(shape=(45,nyears,12))
	#ndviAvg=np.zeros(shape=(nyears,12))
	#ndviMed=np.zeros(shape=(nyears,12))
	ndviAll=-9999*np.ones(shape=(45,vlen,hlen))
	eviAll=-9999*np.ones(shape=(45,vlen,hlen))
	Mask=np.ones(shape=(45,vlen,hlen)) 
	ndwiAll=np.zeros(shape=(45,vlen,hlen))
	####################
	k=-1
	d=-1
	i16N=0
	i15N=0
	for j in range(n_images):
	
		monthAll[j]=str(images['features'][j]['id'][20:30])[5:7]
	
		if monthAll[j]!=monthAll[j-1] and j!=0:
			#if monthAll[j-1]!=6 and monthAll[j-1]!=7 and monthAll[j-1]!=8 and monthAll[j-1]!=9:
			if monthAll[j-1]!=6:
				continue
			for v in range(vlen):
				for h in range(hlen):
					for d in range(45):
						if Mask[d,v,h]==0:
							ndviMonthAvg[y,m,v,h]+=ndviAll[d,v,h]
							eviMonthAvg[y,m,v,h]+=eviAll[d,v,h]
							ndwiMonthAvg[y,m,v,h]+=ndwiAll[d,v,h]
							climoCounter[y,m,v,h]+=1 # number of good days in a month
					ndviClimo[m,v,h]+=ndviMonthAvg[y,m,v,h]
					eviClimo[m,v,h]+=eviMonthAvg[y,m,v,h]
					ndwiClimo[m,v,h]+=ndwiMonthAvg[y,m,v,h]
			d=0
			ndviAll=-9999*np.ones(shape=(45,vlen,hlen))
			eviAll=-9999*np.ones(shape=(45,vlen,hlen))
			Mask=np.ones(shape=(45,vlen,hlen)) 
			ndwiAll=np.zeros(shape=(45,vlen,hlen))
	
		#if monthAll[j]!=6 and monthAll[j]!=7 and monthAll[j]!=8 and monthAll[j]!=9:
		if monthAll[j]!=6:
			d=0
			continue
	
	
	
		# get the scene id
		#scene = images['features'][indexSorted[j]]['key']
		#scene = images['features'][j]['key']
		scene = images['features'][j]['id']
		print scene
		if scene[36:39]=='15N':
			i15N+=1
		if scene[36:39]=='16N':
			i16N+=1
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
				#resolution=dltile['properties']['resolution'],
				#bounds=dltile['properties']['outputBounds'],
				#srs=dltile['properties']['cs_code'],
				resolution=res,
				bands=['ndvi', 'alpha'],
				scales=[[default_range[0], default_range[1], physical_range[0], physical_range[1]]],
				#scales=[[0,16383,-1.0,1.0]],
				data_type='Float32',
				cutline=shape['geometry']
				)
		except:
			print('ndvi: %s could not be retreived' % scene)
			continue

		#######################
		# Get cloud data	  #
		#######################
		#findCloud=-9999*np.ones(shape=(vlen,hlen,4)) 
		#cloudMask=-9999*np.ones(shape=(vlen,hlen)) 
		#
		#band_info_index={}
		#for i in range(len(band_info)):
		#	band_info_index[band_info[i]['name']]=i
		#globals().update(locals())
	
		#try:
		#	default_range = band_info[band_info_index['blue']]['default_range']
		#	data_range = band_info[band_info_index['blue']]['data_range']
		#	blue, meta = dl.raster.ndarray(
		#		scene,
		#		resolution=dltile['properties']['resolution'],
		#		bounds=dltile['properties']['outputBounds'],
		#		srs=dltile['properties']['cs_code'],
		#		bands=['blue', 'alpha'],
		#		scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
		#		data_type='Float32'
		#		)
		#except:
		#	print('blue: %s could not be retreived' % scene)
		#	continue 
	
		#try:
		#	default_range = band_info[band_info_index['red']]['default_range']
		#	data_range = band_info[band_info_index['red']]['data_range']
		#	red, meta = dl.raster.ndarray(
		#		scene, #		resolution=dltile['properties']['resolution'], #		bounds=dltile['properties']['outputBounds'], #		srs=dltile['properties']['cs_code'],
		#		bands=['red', 'alpha'],
		#		scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
		#		data_type='Float32'
		#		)
		#except:
		#	print('red: %s could not be retreived' % scene)
		#	continue 
	
		#try:
		#	default_range = band_info[band_info_index['nir']]['default_range']
		#	data_range = band_info[band_info_index['nir']]['data_range']
		#	nir, meta = dl.raster.ndarray(
		#		scene,
		#		resolution=dltile['properties']['resolution'],
		#		bounds=dltile['properties']['outputBounds'],
		#		srs=dltile['properties']['cs_code'],
		#		bands=['nir', 'alpha'],
		#		scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
		#		data_type='Float32'
		#		)
		#except:
		#	print('nir: %s could not be retreived' % scene)
		#	continue
		#
		#try:
		#	default_range = band_info[band_info_index['swir1']]['default_range']
		#	data_range = band_info[band_info_index['swir1']]['data_range']
		#	swir1, meta = dl.raster.ndarray(
		#		scene,
		#		resolution=dltile['properties']['resolution'],
		#		bounds=dltile['properties']['outputBounds'],
		#		srs=dltile['properties']['cs_code'],
		#		bands=['swir1', 'alpha'],
		#		scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
		#		data_type='Float32'
		#		)
		#except:
		#	print('swir1: %s could not be retreived' % scene)
		#	continue 
	
		#globals().update(locals())
		#findCloud[:,:,0]=blue[:,:,0]
		#findCloud[:,:,1]=red[:,:,0]
		#findCloud[:,:,2]=nir[:,:,0]
		#findCloud[:,:,3]=swir1[:,:,0]
	
		#cloudMask[:,:]=ltk_cloud_mask(findCloud)
	
		try:
			default_range = band_info[band_info_index['alpha']]['default_range']
			data_range = band_info[band_info_index['alpha']]['data_range']
			cloudMask, meta = dl.raster.ndarray(
				scene,
				resolution=res,
				bands=['visual_cloud_mask', 'alpha'],
				scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
				#scales=[[0, 65535, 0., 1.]],
				data_type='Float32',
				cutline=shape['geometry']
				)
		except:
			print('swir1: %s could not be retreived' % scene)
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
		#for v in range(vlen):
		#	for h in range(hlen):
		#if cloudMask[v,h]==3 and v<600:
		#	cloudMask[v,h]=1
		#else:
		#		cloudMask[v,h]=swap[cloudMask[v,h]]
	
		# take out days with too many clouds
		#cloudMask = arrCloud[:, :, 0] == 0 # True=good False=bad
		if np.sum(cloudMask)>0.80*(vlen*hlen):
			print 'clouds: continued', np.round(np.sum(cloudMask)/(vlen*hlen),3)
			continue		
	
		maskforAlpha = arrNDVI[:, :, 1] == 0 
		
		if np.amin(maskforAlpha)==1:
			print 'bad: continued'
			continue
	
		k+=1
		d+=1
		
		###############################################
		# time
		############################################### 
		
		xtime.append(str(images['features'][j]['id'][20:30]))
		#xtime.append(str(images['features'][j]['properties']['acquired'][0:10]))
		date=xtime[k]
		year=int(xtime[k][0:4])
		if k==0:
			startyear=year
			#nyears=2018-startyear
		y=int(year-startyear)
		month=int(xtime[k][5:7])
		m=int(month-1)
		day=xtime[k][8:10]
		dayOfYear=(float(month)-1)*30+float(day)
		plotYear[y,m,d]=year+dayOfYear/365.0
		
		###############################################
		# Back to NDVI
		############################################### 
	
		print date, k, np.round(np.sum(cloudMask)/(vlen*hlen),3), d
		sys.stdout.flush()
		#cloudMask = arrCloud[:, :, 0] != 0 
		#cloudMask = arrCloud[:, :, 1] == 0 #for desert
	
		for v in range(vlen):
			for h in range(hlen):
				if cloudMask[v,h]==0 and maskforAlpha[v,h]==0 and countyMask[v,h]==0: # and oceanMask[v,h]==0:
					Mask[d,v,h]=0
		

		if np.amin(Mask[d])==1:
			print 'bad: continued'
			k-=1
			d-=1
			continue

		#if k==1:
		#	exit()
		#Mask[:,:,k]=1-Mask[:,:,k]
		#Mask[:,:,k]=ndimage.binary_dilation(Mask[:,:,k],iterations=3)
		#Mask[:,:,k]=1-Mask[:,:,k]
		#cloudMask[:,:,k]=1-cloudMask[:,:,k]
		#cloudMask[:,:,k]=ndimage.binary_dilation(Mask[:,:,k],iterations=3)
		#cloudMask[:,:,k]=1-cloudMask[:,:,k]
	
		if makePlots:
			if not os.path.exists(wd+'figures/'+country+'/'+sName+'/'+cName):
				os.makedirs(wd+'figures/'+country+'/'+sName+'/'+cName)
			masked_ndvi = np.ma.masked_array(arrNDVI[:, :, 0], Mask[d,:,:])
			plt.figure(figsize=[10,10])
			plt.imshow(masked_ndvi, cmap=my_cmap, vmin=-.4, vmax=.9)
			#plt.imshow(masked_ndvi, cmap='jet')#, vmin=0, vmax=65535)
			plt.title('NDVI: '+cName+', '+sName+', '+str(date), fontsize=20)
			cb = plt.colorbar()
			cb.set_label("NDVI")
			plt.savefig(wd+'figures/'+country+'/'+sName+'/'+cName+'/ndvi_'+str(date)+'_'+str(k)+'.pdf')
			plt.clf() 
	
		ndviAll[d,:,:]=np.ma.masked_array(arrNDVI[:,:,0],Mask[d,:,:])
	
		###############################################
		# Cloud
		###############################################
	
		if makePlots:
			plt.clf()
			plt.figure(figsize=[10,10])
			plt.imshow(np.ma.masked_array(cloudMask,Mask[d,:,:]), cmap='gray', vmin=0, vmax=1)
			plt.title('Cloud: '+cName+', '+sName+', '+str(date), fontsize=20)
			cb = plt.colorbar()
			cb.set_label("Cloud")
			plt.savefig(wd+'figures/'+country+'/'+sName+'/'+cName+'/cloud_'+str(date)+'_'+str(k)+'.pdf')
			plt.clf()
			
	
		###############################################
		# EVI
		###############################################
		try:
			#default_range= band_info[band_info_index['evi']]['default_range']
			#physical_range = band_info[band_info_index['evi']]['physical_range']
			arrEVI, meta = dl.raster.ndarray(
				scene,
				resolution=res,
				bands=['evi', 'alpha'],
				#scales=[[default_range[0], default_range[1], physical_range[0], physical_range[1]]],
				scales=[[0,2**16,-1.,1.]],
				#scales=[[0,16383,-1.0,1.0]],
				data_type='Float32',
				cutline=shape['geometry']
				)
		except:
			print('evi: %s could not be retreived' % scene)
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
			plt.savefig(wd+'figures/'+country+'/'+sName+'/'+cName+'/evi_'+str(date)+'_'+str(k)+'.pdf')
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
				resolution=res,
				bands=['nir', 'alpha'],
				scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
				#scales=[[0,2**14,-1.,1.]],
				data_type='Float32',
				cutline=shape['geometry']
				)
		except:
			print('nir: %s could not be retreived' % scene)
			continue
	
		nirM=np.ma.masked_array(nir[:,:,0],Mask[d,:,:])
		
		try:
			default_range = band_info[band_info_index['green']]['default_range']
			data_range = band_info[band_info_index['green']]['physical_range']
			green, meta = dl.raster.ndarray(
				scene,
				resolution=res,
				bands=['green', 'alpha'],
				scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
				#scales=[[0,2**14,-1.,1.]],
				data_type='Float32',
				cutline=shape['geometry']
				)
		except:
			print('green: %s could not be retreived' % scene)
			continue
		  
		greenM=np.ma.masked_array(green[:,:,0],Mask[d,:,:])
	
		for v in range(vlen):
			for h in range(hlen):
				ndwiAll[d,v,h] = (green[v,h,0]-nir[v,h,0])/(nir[v,h,0]+green[v,h,0]+1e-9)
			#ndwiAll[v,h,k] = (greenM[v,h]-nirM[v,h])/(nirM[v,h]+greenM[v,h]+1e-9)
		#masked_ndwi = np.ma.masked_array(ndwiAll[:,:,k], Mask[:,:,k])
	
		if makePlots:
			masked_ndwi = np.ma.masked_array(ndwiAll[d,:,:], Mask[d,:,:])
			plt.figure(figsize=[10,10])
			plt.imshow(masked_ndwi, cmap='jet', vmin=-1, vmax=1)
			plt.title('NDWI:' +cName+', '+sName+', '+str(date), fontsize=20)
			cb = plt.colorbar()
			cb.set_label("NDWI")
			plt.savefig(wd+'figures/'+country+'/'+sName+'/'+cName+'/ndwi_'+str(date)+'_'+str(k)+'.pdf')
			plt.clf()
		
		###############################################
		# Visual
		###############################################
		ids = [f['id'] for f in images['features']]
	
		if makePlots:
			arr, meta = dl.raster.ndarray(
				scene,
				resolution=res,
				bands=['red', 'green', 'blue', 'alpha'],
				scales=[[0,4000], [0, 4000], [0, 4000], None],
				data_type='Byte',
				cutline=shape['geometry']
				)
	
			plt.figure(figsize=[10,10])
			plt.imshow(arr)
			plt.title('visual')
			plt.savefig(wd+'figures/'+country+'/'+sName+'/'+cName+'/visual_'+str(date)+'_'+str(k)+'.pdf')
	
		#if k==2:
		#	exit()
		
	########################
	# Save variables	   #
	######################## 
	
	if not os.path.exists(r'../saved_vars/'+sName+'/'+cName):
		os.makedirs(r'../saved_vars/'+sName+'/'+cName)
			 
	#np.save(wd+'saved_vars/'+sName+'/'+cName+'/ndwiAll',ndwiAll) 
	##np.save(wd+'saved_vars/'+sName+'/'+cName+'/Mask',Mask)
	##np.save(wd+'saved_vars/'+sName+'/'+cName+'/oceanMask',oceanMask)
	#np.save(wd+'saved_vars/'+sName+'/'+cName+'/plotYear',plotYear)
	#np.save(wd+'saved_vars/'+sName+'/'+cName+'/n_good_days',int(k))
	##np.save(wd+'saved_vars/'+sName+'/'+cName+'/month',month)
	##np.save(wd+'saved_vars/'+sName+'/'+cName+'/year',year)
	##np.save(wd+'saved_vars/'+sName+'/'+cName+'/arrClas',arrClas)
	#np.save(wd+'saved_vars/'+sName+'/'+cName+'/ndviAll',ndviAll)
	#np.save(wd+'saved_vars/'+sName+'/'+cName+'/eviAll',eviAll)
		
	np.save(wd+'saved_vars/'+sName+'/'+cName+'/ndviClimoUnprocessed',ndviClimo)
	np.save(wd+'saved_vars/'+sName+'/'+cName+'/eviClimoUnprocessed',eviClimo)
	np.save(wd+'saved_vars/'+sName+'/'+cName+'/ndwiClimoUnprocessed',ndwiClimo)
	np.save(wd+'saved_vars/'+sName+'/'+cName+'/climoCounterUnprocessed',climoCounter)
	np.save(wd+'saved_vars/'+sName+'/'+cName+'/ndviMonthAvgUnprocessed',ndviMonthAvg)
	np.save(wd+'saved_vars/'+sName+'/'+cName+'/eviMonthAvgUnprocessed',eviMonthAvg) #for tile in range(len(dltiles['features'])):
	np.save(wd+'saved_vars/'+sName+'/'+cName+'/ndwiMonthAvgUnprocessed',ndwiMonthAvg)
	np.save(wd+'saved_vars/'+sName+'/'+cName+'/countyMask',countyMask)
	
	print i15N, i16N
#	tile=30
#	dltile=dltiles['features'][tile]
#	print len(dltiles['features'])
#tile_function(dltile,makePlots)   
	
	
#for i in range(len(dltiles['features'])):
#	## Check in the bucket
#	## gsutil ls
#	if not os.path.exists(r'../saved_vars/'+str(lonlist[i])+'_'+str(latlist[i])+'/ndviAll'):
#		dltile=dltiles['features'][i]
#		tile_function(dltile)
#		
		
		
		
		
		
		
		
		
		
		

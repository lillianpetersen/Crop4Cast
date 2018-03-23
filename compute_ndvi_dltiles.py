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
import datetime
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
wdvars='/Users/lilllianpetersen/saved_vars/'
wdfigs='/Users/lilllianpetersen/figures/'
wddata='/Users/lilllianpetersen/data/'

countylats=np.load(wdvars+'county_lats.npy')
countylons=np.load(wdvars+'county_lons.npy')
countyName=np.load(wdvars+'countyName.npy')
stateName=np.load(wdvars+'stateName.npy')

#country='Ethiopia'
countryNum='42'

start='2013-01-01'
end='2018-01-03'
nyears=5
nmonths=12
makePlots=False
print 'makePlots=',makePlots
padding = 0
res=120

badarrays=0

f=open(wddata+'africa_latlons.csv')
for line in f:
	tmp=line.split(',')
	if tmp[0]==countryNum:
		country=tmp[1]
		print country
		startlat=float(tmp[2])
		startlon=float(tmp[3])
		pixels=int(tmp[4])

dltile=dl.raster.dltile_from_latlon(startlat,startlon,res,pixels,padding)
print '\n\n'
#print country

#fmonths=open(wddata+'max_ndviMonths.csv')
#for line in fmonths:
#	tmp1=line.split(',')
#	if tmp1[0]==countryNum:
#		country=tmp1[1]
#		corrMonth=tmp1[2][:-1]
#		if len(corrMonth)>2:
#			months=corrMonth.split('/')
#			month1=corrMonth[0]
#			month2=corrMonth[1]
#			corrMonth=month1
#
#corrMonth=int(corrMonth)
#
#if corrMonth<=1:
#	print 'continuing: corrMonth=',corrMonth
#else:
#	print 'stopping: corrMonth=',corrMonth
#	exit()
#
#now = datetime.datetime.now()
#start='2018-01-01'
#end=now.strftime("%Y-%m-%d")

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
print pixels
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
k=-1
d=-1
for j in range(n_images):

	monthAll[j]=str(images['features'][j]['id'][25:27])

	if monthAll[j]!=monthAll[j-1] and j!=0:
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

		if not os.path.exists(wdvars+country+'/2018'):
			os.makedirs(wdvars+country+'/2018')		 
		np.save(wdvars+country+'/ndviClimoUnprocessed',ndviClimo)
		np.save(wdvars+country+'/eviClimoUnprocessed',eviClimo)
		np.save(wdvars+country+'/ndwiClimoUnprocessed',ndwiClimo)
		np.save(wdvars+country+'/climoCounterUnprocessed',climoCounter)
		np.save(wdvars+country+'/ndviMonthAvgUnprocessed',ndviMonthAvg)
		np.save(wdvars+country+'/eviMonthAvgUnprocessed',eviMonthAvg) 
		np.save(wdvars+country+'/ndwiMonthAvgUnprocessed',ndwiMonthAvg)

		d=-1
		ndviAll=-9999*np.ones(shape=(140,pixels,pixels))
		eviAll=-9999*np.ones(shape=(140,pixels,pixels))
		Mask=np.ones(shape=(140,pixels,pixels)) 
		ndwiAll=np.zeros(shape=(140,pixels,pixels))


	# get the scene id
	#scene = images['features'][indexSorted[j]]['key']
	#scene = images['features'][j]['key']
	scene = images['features'][j]['id']
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
		k-=1
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

	k+=1
	d+=1
	
	###############################################
	# time
	############################################### 

	#xtime=str(images['features'][j]['id'][64:74]) # MODIS test
	xtime=str(images['features'][j]['id'][20:30]) # old MODIS
	#xtime.append(str(images['features'][j]['properties']['acquired'][0:10])) Landsat
	date=xtime
	year=int(xtime[0:4])
	if k==0:
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

	print date, k, np.round(np.sum(cloudMask)/(pixels*pixels),3), d, np.round(time2-time1,1)
	sys.stdout.flush()
	#cloudMask = arrCloud[:, :, 0] != 0 
	#cloudMask = arrCloud[:, :, 1] == 0 #for desert

	time1=time.time()

	for v in range(pixels):
		for h in range(pixels):
			if cloudMask[v,h]==0 and maskforAlpha[v,h]==0: # and oceanMask[v,h]==0:
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
		if not os.path.exists(wdfigs+sName+'/'+cName):
			os.makedirs(wdfigs+sName+'/'+cName)
		masked_ndvi = np.ma.masked_array(arrNDVI[:, :, 0], Mask[d,:,:])
		plt.figure(figsize=[10,10])
		plt.imshow(masked_ndvi, cmap=my_cmap, vmin=-.4, vmax=.9)
		#plt.imshow(masked_ndvi, cmap='jet')#, vmin=0, vmax=65535)
		plt.title('NDVI: '+cName+', '+sName+', '+str(date), fontsize=20)
		cb = plt.colorbar()
		cb.set_label("NDVI")
		plt.savefig(wdfigs+sName+'/'+cName+'/ndvi_'+str(date)+'_'+str(k)+'.jpg')
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
		plt.savefig(wdfigs+sName+'/'+cName+'/cloud_'+str(date)+'_'+str(k)+'.jpg')
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
		k-=1
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
		plt.savefig(wdfigs+sName+'/'+cName+'/evi_'+str(date)+'_'+str(k)+'.jpg')
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
		k-=1
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
		k-=1
		d-=1
		continue
	  
	greenM=np.ma.masked_array(green[:,:,0],Mask[d,:,:])

	for v in range(pixels):
		for h in range(pixels):
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
		plt.savefig(wdfigs+sName+'/'+cName+'/ndwi_'+str(date)+'_'+str(k)+'.jpg')
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
		plt.savefig(wdfigs+sName+'/'+cName+'/visual_'+str(date)+'_'+str(k)+'.jpg')

	if makePlots:
		if k==5:
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

if not os.path.exists(wdvars+country):
	os.makedirs(wdvars+country)		 
np.save(wdvars+country+'/ndviClimoUnprocessed',ndviClimo)
np.save(wdvars+country+'/eviClimoUnprocessed',eviClimo)
np.save(wdvars+country+'/ndwiClimoUnprocessed',ndwiClimo)
np.save(wdvars+country+'/climoCounterUnprocessed',climoCounter)
np.save(wdvars+country+'/ndviMonthAvgUnprocessed',ndviMonthAvg)
np.save(wdvars+country+'/eviMonthAvgUnprocessed',eviMonthAvg) 
np.save(wdvars+country+'/ndwiMonthAvgUnprocessed',ndwiMonthAvg)

#for tile in range(len(dltiles['features'])):
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

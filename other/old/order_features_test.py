import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import pickle
import pandas as pd
import math
import sys
from sys import exit
import sklearn
from sklearn import svm
import time
from sklearn.preprocessing import StandardScaler

####################
# Function         #
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
    
####################        

wd='/Users/lilllianpetersen/Google Drive/science_fair/'

#lonAll=[-57.864114, -57.345734, -56.0979, -95.156364, -123.6585, -110.580639, 111.233326]
#latAll= [-13.458213, -12.748814,  -15.6014, 41.185114, 39.3592, 35.772751, 51.158285]
#lon=lonAll[4]
#lat=latAll[4]
#lon=-120.3631
#lat=38.4083
vlen=992
hlen=992
start='2016-01-01'
end='2016-12-31'
nyears=1
country='US'
makePlots=False
padding = 16
pixels = vlen+2*padding
res = 120.0

#compute_ndvi(lon,lat,pixels,start,end,country,makePlots)     

clas=["" for x in range(12)]
clasLong=["" for x in range(255)]
clasDict={}
clasNumDict={}
f=open(wd+'data/ground_data.txt')                                
for line in f:
    tmp=line.split(',')
    clasNumLong=int(tmp[0])
    clasLong[clasNumLong]=tmp[1]
    clasNum=int(tmp[3])
    clas[clasNum]=tmp[2]
    
    clasDict[clasLong[clasNumLong]]=clas[clasNum]
    clasNumDict[clasNumLong]=clasNum
    
    
"""function to compute ndvi and make summary plots
variables: lon, lat, pixels, start, end, country, makePlots
"""

#matches=dl.places.find('united-states_california')
#aoi = matches[0]
#shape = dl.places.shape(aoi['slug'], geom='low')

matches=dl.places.find('united-states_washington')
aoi = matches[0]
shape = dl.places.shape(aoi['slug'], geom='low')

#dltile = dl.raster.dltile_from_latlon(lat, lon, res, valid_pix, padding)
dltiles = dl.raster.dltiles_from_shape(res, vlen, padding, shape)

features=np.zeros(shape=(len(dltiles['features']),nyears,pixels*pixels,6))
target=np.zeros(shape=(len(dltiles['features']),nyears,pixels*pixels))

#features=pickle.load(open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/features','rd'))
#target=pickle.load(open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/target','rd'))

tile=0
dltile=dltiles['features'][tile]
lon=dltile['geometry']['coordinates'][0][0][0]
lat=dltile['geometry']['coordinates'][0][0][1]

images = dl.metadata.search(
    const_id=["MO", "MY"],
    start_time=start,
    end_time=end,
    geom=dltile['geometry'],
    cloud_fraction=0.9,
    limit = 2000
    )

n_images = len(images['features'])
print('Number of image matches: %d' % n_images)
k=n_images
mo = dl.raster.get_bands_by_constellation("MO").keys()
my = dl.raster.get_bands_by_constellation("MY").keys()
avail_bands = set(mo).intersection(my)
print('Available bands: %s' % ', '.join([a for a in avail_bands]))

band_info = dl.raster.get_bands_by_constellation("MO")

#    k=[j['id'][20:30] for j in images['features']]
#timek=[time.mktime(time.strptime(j,"%Y-%m-%d")) for j in k]
#sorted_features=sorted(images['features'],key=lambda d:['id'][20:30])
#sorted_features=sorted(images['features'],key=str.lower(str(g[j for j in range(len(g))])))


dayOfYear=np.zeros(shape=(k+1))
year=np.zeros(shape=(k+1))
month=np.zeros(shape=(k))
day=np.zeros(shape=(k))
plotYear=np.zeros(shape=(k))
xtime=[]
i=-1
for feature in images['features']:
    i+=1
    # get the scene id
    scene = feature['id']
     
    xtime.append(str(images['features'][i]['id'][20:30]))
    date=xtime[i]
    year[i]=xtime[i][0:4]
    month[i]=xtime[i][5:7]
    day[i]=xtime[i][8:10]
    dayOfYear[i]=(float(month[i])-1)*30+float(day[i])
    plotYear[i]=year[i]+dayOfYear[i]/365.0

#    print date, k
#    sys.stdout.flush()
    
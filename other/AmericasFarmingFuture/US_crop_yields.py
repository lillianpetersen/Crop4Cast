############################################
# Read in crop yield data
# Finds lat and lon of counties
############################################
from pylab import *
import csv
from math import sqrt
from sys import exit
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from mpl_toolkits.basemap import Basemap
import os
from scipy.stats import norm
import matplotlib as mpl
from matplotlib.patches import Polygon
import random

# cd Documents/Science_Fair_2017_Crop_Yields/

# the number of everything
nCounties=3143
nCrop=3
nYears=116
nStates=57

#files to open
f=open('data/crop_yields.csv','r')
ft=open('data/crop_totals.csv','r')
floc=open('data/countylatlon.csv')

presentGrowingCounties=pickle.load(open('pickle_files/presentGrowingCounties.p','rb'))

# initialize variables
iCounty=0 # a counter of the number of counties
cIndex=-9999*ones(shape=(nStates,850))
cropYield=-9999.*ones(shape=(nCounties,nYears,nCrop))
cropTotal=-9999.*ones(shape=(nCounties,nYears,nCrop))
countyName=[]
lat=-9999.*ones(shape=(nCounties))
lon=-9999.*ones(shape=(nCounties))
cIndexState={} # give it cIndex, returns state
highTotalYield=zeros(shape=(nCounties,nCrop),dtype=bool)

#                              #
# reads in county lat and lons #
#                              #
for line in floc:
    tmp=line.split(',')
    
    stateID=float(tmp[1])
    countyID=float(tmp[3])
    countyName.append(tmp[2])
    
    s=stateID
    c=countyID
        
    cIndex[s,c]=iCounty
    cIndexState[cIndex[s,c]]=s
    
    lat[cIndex[s,c]]=tmp[5]
    lon[cIndex[s,c]]=tmp[6]
    iCounty+=1
floc.close()
j=-1
countyName=[''  for x in xrange(3143)]

#                          #
# reads in crop yield data #
#                          #
for line in f:
    j+=1
    line=line.translate(None, ',')
    tmp=line.split('""')
    
    if tmp[9]=='OTHER (COMBINED) COUNTIES':
        continue
    
    year=float(tmp[1])
    stateID=float(tmp[6])
    countyID=float(tmp[10])
    crop=tmp[15]
    yd=float(tmp[19])
    
    if crop=='CORN':
        cropID=0
    if crop=='SOYBEANS':
        cropID=1
    if crop=='RICE':
        cropID=2
    
    y=year-1900
    s=stateID
    c=countyID
    cp=cropID
    
    if cIndex[s,c]==-9999:
        continue
    cropYield[cIndex[s,c],y,cp]=yd 
    countyName[int(cIndex[s,c])]=tmp[9]

f.close()

j=0
#                           #
# reads in total yield data #
#                           #
for line in ft:
    j+=1
    line=line.translate(None, ',')
    tmp=line.split('""')
    
    if tmp[9]=='OTHER (COMBINED) COUNTIES':
        continue
    
    year=float(tmp[1])
    stateID=float(tmp[6])
    countyName.append(tmp[9])
    countyID=float(tmp[10])
    crop=tmp[15]
    yd=float(tmp[19])
    
    if crop=='CORN':
        cp=0
    if crop=='SOYBEANS':
        cp=1
    if crop=='RICE':
        cp=2
        
    y=year-1900
    s=stateID
    c=countyID
    
    if cIndex[s,c]==-9999:
        continue
    cropTotal[cIndex[s,c],y,cp]=yd 
ft.close()

for cp in range(nCrop):
    cutoff=.1*(np.amax(cropTotal[:,115,cp]))
    for icity in range(3143):
        total=average(cropTotal[icity,105:116,cp])
        if total>=cutoff:
            highTotalYield[icity,cp]=True     
            
#pickle.dump(highTotalYield,open('pickle_files/highTotalYield.p','wb'))  
        
#pickle.dump(cropTotal,open('pickle_files/cropTotal.p','wb'))

# create the map
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# load the shapefile, use the name 'states'
map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)

ax = plt.gca() # get current axes instance

# collect the state names from the shapefile attributes so we can
# look up the shape obect for a state by it's name
j=0 #counter for how many times loop goes through
g=0 
cmapArray=plt.cm.jet(arange(256))
cmin=0
y1=0
y2=256
year=115
cp=2
cmax=.6*(np.amax(cropTotal[:,115,cp]))
cmin=np.amin(cropTotal[:,115,cp])
#cmax=np.amax(cropYield[:,115,cp])
#cmax=8

for shape_dict in map.states_info:
    seg = map.states[j]
    
    s=int(shape_dict['STATEFP'])
    c=int(shape_dict['COUNTYFP'])
    
    if s==72 or cIndex[s,c]==-9999 or presentGrowingCounties[cIndex[s,c],cp]==0:
        j+=1
        continue

    Yield=average(cropTotal[cIndex[s,c],105:116,cp])
    
    if Yield<=0:
        j+=1
        continue
        
    x=Yield
    y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
    icmap=min(255,int(round(y,1)))
       
    poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
    ax.add_patch(poly)
       
    j+=1
    g+=1
plt.show() 
    
# pickle the values
pickle.dump(lat,open('pickle_files/county_lats.p','wb'))
pickle.dump(lon,open('pickle_files/county_lons.p','wb'))
pickle.dump(cropYield,open('pickle_files/cropYield.p','wb'))
pickle.dump(cropTotal,open('pickle_files/cropTotal.p','wb'))
pickle.dump(cIndex,open('pickle_files/cIndex.p','wb'))
pickle.dump(cIndexState,open('pickle_files/cIndexState.p','wb'))  
pickle.dump(countyName,open('pickle_files/countyName.p','wb'))
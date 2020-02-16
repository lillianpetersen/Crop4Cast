from pylab import *
import csv
from math import sqrt
from sys import exit
import numpy as np
import pickle
import os

countyLat=pickle.load(open('pickle_files/county_lats.p','rb'))
countyLon=pickle.load(open('pickle_files/county_lons.p','rb'))
cmiplat=pickle.load(open('pickle_files/cmiplatall.p','rb'))
cmiplon=pickle.load(open('pickle_files/cmiplonall.p','rb'))
exit()
# initialize variables
nearestStationLat=-9999*ones(shape=(3143))
nearestStationLon=-9999*ones(shape=(3143))
latIndex=-9999*ones(shape=(3143))
lonIndex=-9999*ones(shape=(3143))

f=open('written_files/model_nearest_stations.txt','w')
zero=0

for c in range(3143):
    print c, ' of 3143'
    lat=countyLat[c]
    lon=countyLon[c]
    shortestDist=.15
    shortestDist2=.15
    b=0
    g=0
    for ilat in range(len(cmiplat)):
        modelLat=cmiplat[ilat]
        for ilon in range(len(cmiplon)):
            modelLon=cmiplon[ilon]-360
            
            # find distance from county center to weather station
            dist=abs(sqrt((lon-modelLon)**2+(lat-modelLat)**2))
                
            
            if dist>.1: # only look at the close ones
                b+=1
                continue
            
            if dist<shortestDist: # if its closer than the last closest one
                g+=1
                # make this one the closest one
                shortestDist=dist
                nearestStationLat[c]=modelLat
                nearestStationLon[c]=modelLon
                latIndex[c]=ilat
                lonIndex[c]=ilon
                         
    if g==0:
        zero+=1
        shortestDist=-9999
        nearestStationLat[c]=-9999
        nearestStationLon[c]=-9999
        latIndex[c]=-9999
        lonIndex[c]=-9999
        
    # write the closest station into a csv file      
    f.write(str(nearestStationLat[c])+','+str(nearestStationLon[c])+','+
        str(int(latIndex[c]))+','+str(int(lonIndex[c]))+'\n')

        
f.close()
            
            
            
            
            
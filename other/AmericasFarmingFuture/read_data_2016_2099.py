# the Scientific Python netCDF 3 interface
# http://dirac.cnrs-orleans.fr/ScientificPython/
# from Scientific.IO.NetCDF import NetCDFFile as Dataset
# the 'classic' version of the netCDF4 python interface
# http://code.google.com/p/netcdf4-python/
from netCDF4 import Dataset
from numpy import arange # array module from http://numpy.scipy.org
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylab import *
import matplotlib
import csv
from math import sqrt
from sys import exit
import numpy as np
import os
import pickle
from datetime import datetime
from datetime import timedelta
import sys

TMAX=True
TMIN=False

rcp85=False
rcp45=True

# open a the netCDF file for reading.
if rcp85:
    if TMAX:    
        ncfile = (Dataset('tasmax_rcp85_2092_2099.nc','r'),
            Dataset('tasmax_rcp85_2084_2091.nc','r'),
            Dataset('tasmax_rcp85_2076_2083.nc','r'),
            Dataset('tasmax_rcp85_2068_2075.nc','r'),
            Dataset('tasmax_rcp85_2060_2067.nc','r'),
            Dataset('tasmax_rcp85_2052_2059.nc','r'),
            Dataset('tasmax_rcp85_2044_2051.nc','r'),
            Dataset('tasmax_rcp85_2036_2043.nc','r'),
            Dataset('tasmax_rcp85_2028_2035.nc','r'),
            Dataset('tasmax_rcp85_2020_2027.nc','r'),
            Dataset('tasmax_rcp85_2016_2019.nc','r'))
        t='tmax'
    
    if TMIN:
        ncfile = (Dataset('tasmin_rcp85_2092_2099.nc','r'),
            Dataset('tasmin_rcp85_2084_2091.nc','r'),
            Dataset('tasmin_rcp85_2076_2083.nc','r'),
            Dataset('tasmin_rcp85_2068_2075.nc','r'),
            Dataset('tasmin_rcp85_2060_2067.nc','r'),
            Dataset('tasmin_rcp85_2052_2059.nc','r'),
            Dataset('tasmin_rcp85_2044_2051.nc','r'),
            Dataset('tasmin_rcp85_2036_2043.nc','r'),
            Dataset('tasmin_rcp85_2028_2035.nc','r'),
            Dataset('tasmin_rcp85_2020_2027.nc','r'),
            Dataset('tasmin_rcp85_2016_2019.nc','r'))
        t='tmin'
        
if rcp45:
    if TMAX:    
        ncfile = (Dataset('tasmax_rcp45_2092_2099.nc','r'),
            Dataset('tasmax_rcp45_2084_2091.nc','r'),
            Dataset('tasmax_rcp45_2076_2083.nc','r'),
            Dataset('tasmax_rcp45_2068_2075.nc','r'),
            Dataset('tasmax_rcp45_2060_2067.nc','r'),
            Dataset('tasmax_rcp45_2052_2059.nc','r'),
            Dataset('tasmax_rcp45_2044_2051.nc','r'),
            Dataset('tasmax_rcp45_2036_2043.nc','r'),
            Dataset('tasmax_rcp45_2028_2035.nc','r'),
            Dataset('tasmax_rcp45_2020_2027.nc','r'),
            Dataset('tasmax_rcp45_2016_2019.nc','r'))
        t='tmax'
    
    if TMIN:
        ncfile = (Dataset('tasmin_rcp45_2092_2099.nc','r'),
            Dataset('tasmin_rcp45_2084_2091.nc','r'),
            Dataset('tasmin_rcp45_2076_2083.nc','r'),
            Dataset('tasmin_rcp45_2068_2075.nc','r'),
            Dataset('tasmin_rcp45_2060_2067.nc','r'),
            Dataset('tasmin_rcp45_2052_2059.nc','r'),
            Dataset('tasmin_rcp45_2044_2051.nc','r'),
            Dataset('tasmin_rcp45_2036_2043.nc','r'),
            Dataset('tasmin_rcp45_2028_2035.nc','r'),
            Dataset('tasmin_rcp45_2020_2027.nc','r'),
            Dataset('tasmin_rcp45_2016_2019.nc','r'))
        t='tmin'
    
# read the data in variable named 'data'.
print 'read lat:'
lat = ncfile[0].variables['lat'][:]
print 'read lon'
lon = ncfile[0].variables['lon'][:]
print 'open nearest station file:'
f=open('../written_files/model_nearest_stations.txt','r')

nCities=3143

modelData45=-9999*ones(shape=(31,12,85,3143),dtype=np.float16)
cmiplat=zeros(shape=(nCities),dtype=np.float16)
cmiplon=zeros(shape=(nCities),dtype=np.float16)

for files in range(11):
    f.close()
    f=open('../written_files/model_nearest_stations.txt','r')
    print '\n\n'+t; sys.stdout.flush()
    print str(files) +' of 10'; sys.stdout.flush()
    time=ncfile[files].variables['time']
            
    icity=-1
    for cityline in f:
        icity+=1 
        print icity,'of',nCities; sys.stdout.flush()
        
        tmp=cityline.split(',')
        lat=tmp[0]
        lon=tmp[1]
        ilat=tmp[2]
        ilon=tmp[3]
        cmiplat[icity]=lat
        cmiplon[icity]=lon
        
        if int(float(lat))==-9999:
            print 'lat='+lat+': continued'; sys.stdout.flush()
            continue   
        if TMAX:
            tmax=ncfile[files].variables['air_temperature'][:,ilat,ilon]
        if TMIN:
            tmin=ncfile[files].variables['air_temperature'][:,ilat,ilon]
        
        for iday in range(len(time)):
            date_format = "%m/%d/%Y"
            d = timedelta(days=int(time[iday]))
            a = datetime.strptime('1/1/1900', date_format)
            date=a+d
            year=date.year
            month=date.month
            day=date.day
            
            y=year-2016
            m=month-1
            d=day-1
            
            if TMAX:    
                modelData45[d,m,y,icity]=(tmax[iday]-273.15)*1.8+32.0 # convert K to F
            if TMIN:
                modelData45[d,m,y,icity]=(tmin[iday]-273.15)*1.8+32.0 # convert K to F 
     
print cmiplat
print cmiplon
print np.shape(cmiplat)

pickle.dump(cmiplat,open('cmiplatall45.p','wb'))
pickle.dump(cmiplon,open('cmiplonall45.p','wb'))
pickle.dump(modelData45,open('modelData_all_'+t+'_rcp45.p','wb'))
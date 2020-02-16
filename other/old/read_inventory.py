###############################################
# The First Code
# Goes the invintory file and picks the cities over a certain number of years
# writes all of the info about those cities into a text document
# plots the cities 
###############################################
from pylab import *
import csv
from math import sqrt
from sys import exit
import numpy
from mpl_toolkits.basemap import Basemap
import os

#cd \Users\lilli_000\Documents\Science_Fair_2016_Climate_Change\code

## Number of Years Wanted ##
years=65

###############################################
# Read in Station Names
###############################################
filename='data/noaa_daily_data/ghcnd-stations.txt'
openStations=open(filename,'r')

cityName={}  #initialize a dictionary

for line in openStations:
    tmp=line[0:85]
    station=tmp[0:11]
    cityName[station]=tmp[41:70]
       
###############################################
# Read in Start and End Date Data
###############################################

#the inventory file to read in
fileOpen=open('data/noaa_daily_data/ghcnd-inventory.txt','r') 

# a file I am going to write with Tmax for cities over however many years
f=open('written_files/tmax_cities_over_'+str(years)+'_years_no_usc.txt','w')

# a file I am going to write with precip for cities over however many years
fprecip=open('written_files/precip_cities_over_'+str(years)+'_years_no_usc.txt','w')

#                      #
# initialize variables #
#                      #
k=0 #a counter of how many times the loop goes through
maxk=571433 # the maximum number of stations
lengthAbove50=0  #9969
lengthAbove80=0  #3670
lengthAbove100=0 #2493
lengthAbove120=0 #939
lengthAbove130=0
startYear=zeros(shape=(maxk))
endYear=zeros(shape=(maxk))
length=zeros(shape=(maxk))
var=zeros(shape=(maxk))
lat=zeros(shape=(maxk))
lon=zeros(shape=(maxk))
latPlot=zeros(shape=(50000))
lonPlot=zeros(shape=(50000))
kPlot=0
kPrecipPlot=0
latPrecipPlot=zeros(shape=(50000))
lonPrecipPlot=zeros(shape=(50000))
longCity=zeros(shape=(maxk),dtype=bool)


#                                                                              #
# read in each the lat, lon, station name, startYear, endYear for each station #
#                                                                              #
for line in fileOpen: # loop through every station
    tmp=line[0:45]
    station=tmp[0:11]
    lat[k]=tmp[11:20]
    lon[k]=tmp[21:30]
    variable=tmp[31:35]
    startYear[k]=float(tmp[36:40])
    endYear[k]=float(tmp[41:45])
    #print k,startYear[k],tmp[30:40]
    
    
    if variable=='TMAX':
        var[k]=0
    elif variable=='TMIN':
        var[k]=1
    elif variable=='PRCP':
        var[k]=2
    elif variable=='SNOW':
        var[k]=3
    else:
        var[k]=-1
        
    length[k]=endYear[k]-startYear[k] # find how many years of data the station has
    
    ## find how many stations have data over a certain number of years ##
    if var[k]==0: 
        if length[k]>=50:
            lengthAbove50+=1
        if length[k]>=80:
            lengthAbove80+=1
        if length[k]>=100:
            lengthAbove100+=1
        if length[k]>=120:
            lengthAbove120+=1
        if length[k]>=130:
            lengthAbove130+=1
            
    # find the cities that have atleast years worth of data, the var is Tmax, and ends after 2010
    if length[k]>=years and var[k]==0 and endYear[k]>2010: 
        longCity[k]=True
        latPlot[kPlot]=lat[k] # put the latitude of the city into an array
        lonPlot[kPlot]=lon[k] # put the longitude of the city into an array
        kPlot+=1 # a counter of how many cities
        # write the station, lat, lon, var, startYear, endYear, and place where located into a document
        f.write(tmp+' '+cityName[station]+'\n')
     
    # find the cities that have atleast years worth of data, the var is precip, and ends after 2010    
    if length[k]>=years and var[k]==2 and endYear[k]>2010:
        # write the station, lat, lon, var, startYear, endYear, and place where located into a document
        fprecip.write(tmp+' '+cityName[station]+'\n')
        latPrecipPlot[kPrecipPlot]=lat[k] # put the latitude of the city into an array
        lonPrecipPlot[kPrecipPlot]=lon[k] # put the longitude of the city into an array
        kPrecipPlot+=1 # a counter of how many cities
                   
    k+=1 # add one to the counter of how many times the loop has gone through
f.close()
fprecip.close()
print kPlot # print number of cities written into the Tmax file
print kPrecipPlot # print number of cities written into the precip file

figure(1,figsize=(20, 20))
# plot the cities
m = Basemap(llcrnrlat=-60,urcrnrlat=80,\
    llcrnrlon=-180,urcrnrlon=180,lat_ts=50,resolution='i',area_thresh=10000)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='#BDFCC9')
x,y=m(lonPlot,latPlot)
#x,y=-74.4242,39.3792
plotHandle = m.scatter(x,y,c=lonPrecipPlot,s=80,cmap='gist_rainbow',
    edgecolor='none', vmin=-90,vmax=90,zorder=10) 
m.colorbar(plotHandle)
title('Cities with Precip records over '+str(years)+' years')
savefig('figures/many_cities.png',dpi=500)
show()
################################################
# Make movie of Yield summer avg over time
################################################
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

###############################################
# Functions
###############################################
def Avg(x):   
    '''function to average'''
    xAvg=0.
    for k in range(len(x)):
        xAvg=xAvg+x[k]
    xAvg=xAvg/(k+1)
    return xAvg 
    
    
###############################################
# Variables
###############################################
futureYield=pickle.load(open('pickle_files/futureYield.p','rb'))
# futureYield dimensions = (nCities,nPredictor,nyears,nScen,nCrop)
# nPredictor  0=Summer avg, 1=Heat Waves, 2=KDD
cropYield=pickle.load(open('pickle_files/cropYield.p','rb'))
# cropYield dimensions = (nCities,year,cp)
cIndex=pickle.load(open('pickle_files/cIndex.p','rb'))
cIndexState=pickle.load(open('pickle_files/cIndexState.p','rb'))
BBcounty=pickle.load(open('pickle_files/BBcounty.p','rb'))
SummerAvg=pickle.load(open('pickle_files/SummerAvg.p','rb'))
KDD=pickle.load(open('pickle_files/KDD.p','rb'))
HeatWaves=pickle.load(open('pickle_files/HeatWaves.p','rb'))
presentGrowingCounties=pickle.load(open('pickle_files/presentGrowingCounties.p','rb'))
KDDFuture=pickle.load(open('pickle_files/KDDFuture.p','rb'))
HeatWavesFuture=pickle.load(open('pickle_files/HeatWavesFuture.p','rb'))
SummerAvgFuture=pickle.load(open('pickle_files/SummerAvgFuture.p','rb'))
highTotalYield=pickle.load(open('pickle_files/highTotalYield.p','rb'))

goodfuture=zeros(shape=(3143,3),dtype=bool)
goodpast=zeros(shape=(3143,3),dtype=bool)
goodpastyears=zeros(shape=(3143,116,3),dtype=bool)

avgFutureYield=zeros(shape=(100,3))
avgCropYield=zeros(shape=(116,3)) 
nCrop=3  
cropTitle=('Corn','Soybeans','Rice')

for icity in range(3143):
    for y in range(16,100):
        for cp in range(nCrop):
            if futureYield[icity,0,y,0,cp]>0 and presentGrowingCounties[icity,cp]==1 and isnan(futureYield[icity,0,y,0,cp])==False:
                goodfuture[icity,cp]=True
    for y in range(70,116):
        for cp in range(nCrop):
            if cropYield[icity,y,cp]>0 and highTotalYield[icity,cp]==True:
                    goodpast[icity,cp]=True
                    goodpastyears[icity,y,cp]=True
       
###############################################
# Plot Yield 1970-2015
###############################################
for cp in range(3):
    cmapArray=plt.cm.jet(arange(256))
    cmin=0
    y1=0
    y2=256
    year=115
    cmax=.8*(np.amax(cropYield[:,105:116,cp]))
    
    for year in range(70,116):
        print year+1900
        
        figure(1,figsize=(9,7))
        show()
        
        map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
        
        # load the shapefile, use the name 'states'
        map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
        
        ax = plt.gca() # get current axes instance
        j=-1
        
        for shape_dict in map.states_info:
            j+=1
            seg = map.states[j]
            
            s=int(shape_dict['STATEFP'])
            c=int(shape_dict['COUNTYFP'])
            
            if s==72 or cIndex[s,c]==-9999:
                continue
        
            Yield=cropYield[cIndex[s,c],year,cp]
            
            if Yield<=0:
                continue
                
            x=Yield
            y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
            icmap=min(255,int(round(y,1)))
            
            poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
            ax.add_patch(poly)
            
        title('Bushels/Acre of '+cropTitle[cp]+' Grown')
        text(3500000,2500000,year+1900,fontsize=25)
        savefig('final_figures/crop_yield_every_year/'+cropTitle[cp]+'/'+cropTitle[cp]+'_yield_'+str(year+1900))
        plt.show()      
        clf()
    
###############################################
# Plot Yield 2016-2100
###############################################
for cp in range(nCrop):
    cmapArray=plt.cm.jet(arange(256))
    cmin=0
    y1=0
    y2=256
    cmax=.8*(np.amax(cropYield[:,105:116,cp]))
    
    for year in range(16,100):
        print year+2000
        
        figure(1,figsize=(9,7))
        show()
        
        map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
        
        # load the shapefile, use the name 'states'
        map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
        
        ax = plt.gca() # get current axes instance
        j=-1
        
        for shape_dict in map.states_info:
            j+=1
            seg = map.states[j]
            
            s=int(shape_dict['STATEFP'])
            c=int(shape_dict['COUNTYFP'])
            
            if s==72 or cIndex[s,c]==-9999:
                continue
        
            Yield=mean(futureYield[cIndex[s,c],0:2,year,0,cp])
            
            if Yield<=0 or isnan(Yield)==True:
                continue
                
            x=Yield
            y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
            icmap=min(255,int(round(y,1)))
            
            poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
            ax.add_patch(poly)
            
        title('Bushels/Acre of '+cropTitle[cp]+' Grown in '+str(year+2000))
        text(3500000,2500000,year+2000,fontsize=25)
        savefig('final_figures/crop_yield_every_year/'+cropTitle[cp]+'/'+cropTitle[cp]+'_yield_'+str(year+2000))
        plt.show() 
        clf()

###############################################
# Plot Summer Avg Temp
###############################################
cmapArray=plt.cm.jet(arange(256))
cmin=74
y1=0
y2=256
cmax=98

for year in range(70,115):
    b=0
    print year+1900
    
    figure(1,figsize=(9,7))
    show()
    
    map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    
    # load the shapefile, use the name 'states'
    map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
    
    ax = plt.gca() # get current axes instance
    j=-1
    
    for shape_dict in map.states_info:
        j+=1
        seg = map.states[j]
        
        s=int(shape_dict['STATEFP'])
        c=int(shape_dict['COUNTYFP'])
        
        if s==72 or cIndex[s,c]==-9999:
            b+=1
            continue
    
        Temp=SummerAvg[cIndex[s,c],year]
        
        if Temp<=0:
            b+=1
            continue
            
        x=Temp
        y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
        icmap=min(255,int(round(y,1)))
        
        poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
        ax.add_patch(poly)
        
    title('Summer Average Temperature')
    text(3500000,2500000,year+1900,fontsize=25)
    savefig('final_figures/summer_avg_every_year/summer_'+str(year+1900))
    plt.show() 
    clf()

###############################################
# Plot Summer Avg Temp 2016-2100
###############################################
cmapArray=plt.cm.jet(arange(256))
cmin=74
y1=0
y2=256
cmax=98

for year in range(16,100):
    b=0
    print year+2000
    
    figure(1,figsize=(9,7))
    show()
    
    map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    
    # load the shapefile, use the name 'states'
    map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
    
    ax = plt.gca() # get current axes instance
    j=-1
    
    for shape_dict in map.states_info:
        j+=1
        seg = map.states[j]
        
        s=int(shape_dict['STATEFP'])
        c=int(shape_dict['COUNTYFP'])
        
        if s==72 or cIndex[s,c]==-9999:
            b+=1
            continue
    
        Temp=SummerAvgFuture[cIndex[s,c],year]
        
        if Temp<=0 or isnan(Temp)==True:
            b+=1
            continue
            
        x=Temp
        y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
        icmap=min(255,int(round(y,1)))
        
        poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
        ax.add_patch(poly)
        
    title('Summer Average Temperature in '+str(year+2000))
    text(3500000,2500000,year+2000,fontsize=25)
    savefig('final_figures/summer_avg_every_year/summer_'+str(year+2000))
    plt.show() 
    clf()
  
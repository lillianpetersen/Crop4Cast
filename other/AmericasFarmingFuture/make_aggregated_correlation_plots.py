################################################
# Makes the aggregated plots
# Makes the slope plots
################################################
from pylab import *
import csv
import math
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
def reverse_colormap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []  
    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r 
    
###############################################
# Pickle Files
###############################################    
cityFile='tmax_cities_over_45_years'
cIndexState=pickle.load(open('pickle_files/cIndexState.p','rb'))
Corr=pickle.load(open('pickle_files/Corr.p','rb'))
maxCorr=pickle.load(open('pickle_files/maxCorr.p','rb'))
iplotCorr=pickle.load(open('pickle_files/iplotCorr.p','rb'))
maxCorr2=pickle.load(open('pickle_files/maxCorr2.p','rb'))
iplotCorr2=pickle.load(open('pickle_files/iplotCorr2.p','rb'))
slope=pickle.load(open('pickle_files/slope.p','rb'))
R_squared=pickle.load(open('pickle_files/'+cityFile+'/R2.p','rb'))
lat=pickle.load(open('pickle_files/'+cityFile+'/lat.p','rb'))
lon=pickle.load(open('pickle_files/'+cityFile+'/lon.p','rb'))
station=pickle.load(open('pickle_files/'+cityFile+'/station.p','rb'))
goodCity=pickle.load(open('pickle_files/'+cityFile+'/goodCity.p','rb'))
cIndex=pickle.load(open('pickle_files/cIndex.p','rb'))
presentGrowingCounties=pickle.load(open('pickle_files/presentGrowingCounties.p','rb'))
highTotalYield=pickle.load(open('pickle_files/highTotalYield.p','rb'))
goodCity3=pickle.load(open('pickle_files/goodCity3.p','rb'))

###############################################
# Define Variables
###############################################
v=0

lat=lat[goodCity[v,:]]
lon=lon[goodCity[v,:]]
numGoodCities=sum(goodCity[v,:])
dataYield=ones(shape=(3143,2),dtype=bool)
cropTitle=('Corn','Soybeans','Rice')

#var=('Time','Normalized_Time','Growing_Degree_Days','Killing_Degree_Days','Frost_Nights',
#    'Season_Frost_Nights','Tmax_Avg','Tmin_Avg','Spring_Avg','Summer_Avg','Tropical_Nights',
#    'Tmax_Heat_Waves','Tmin_Heat_Waves','Tmax_Warm_Days','Tmin_Warm_Days')

var=('Time','Normalized_Time','Killing_Degree_Days','Summer_Avg','Heat_Waves')
titleVar=('Time','Normalized Time','Killing Degree Days','Summer Avg Temp','Heat Waves')
   
cp=0
cmapArray=plt.cm.hot(arange(256))
cmin=-0.7
cmax=0
y1=0
y2=255

MinMaxArray=ones(shape=(2,3))

b=0
significant=0
highlySignificant=0
countyCounter=0
###############################################
# Plot
###############################################
MinMaxArray=zeros(shape=(2,1))
goodCity2=zeros(shape=(3143,3),dtype=bool)

for iplot in range(4,5):
    print iplot
    
    figure(5,figsize=(9,7))
    show()
    
    for icity in range(3143):
        if Corr[iplot,icity,cp]!=0:
            goodCity2[icity,cp]=True
    
    j=-1
    
    subPlot1 = plt.axes([.5,.25,.475,.5])
    MinMaxArray[0,0]=-0.7
    MinMaxArray[1,0]=0
    
    plt.imshow(MinMaxArray,cmap='hot')
    plt.colorbar()
    
    # create the map
    subPlot1 = plt.axes([0.1,0,0.8,1])
    m = Basemap(llcrnrlon=-107,llcrnrlat=27,urcrnrlon=-71,urcrnrlat=49,
    lat_ts=50,resolution='i',area_thresh=10000)
            
    # load the shapefile, use the name 'states'
    m.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
    
    ax = plt.gca() # get current axes instance
    
    for shape_dict in m.states_info:
        j+=1
        seg = m.states[j]
        
        s=int(shape_dict['STATEFP'])
        c=int(shape_dict['COUNTYFP'])
        
        if s==72 or cIndex[s,c]==-9999:
            continue
            
        if goodCity2[cIndex[s,c],cp]==False:
            continue
            
        x=Corr[iplot,cIndex[s,c],cp]
        if math.isnan(x)==True:
            b+=1
            goodCity2[cIndex[s,c],cp]=False
            continue
        countyCounter+=1
        if x<=-.49:
            significant+=1
        if x<=-.59:
            highlySignificant+=1
        y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
        icmap=min(255,int(round(y,1)))
        
         
        poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
        ax.add_patch(poly)
    #print titleVar[iplot],round(mean(Corr[iplot,goodCity2[:,cp],cp]),2),significant, round((float(significant)/75.)*100,2)
    #print titleVar[iplot],round(mean(Corr[iplot,goodCity2[:,cp],cp]),2),highlySignificant, round((float(highlySignificant)/75.)*100,2)
    title('Correlation of '+cropTitle[cp]+' and '+titleVar[iplot]+', Avg Corr= '+str(round(mean(Corr[iplot,goodCity2[:,cp],cp]),2)))
    savefig('final_figures/correlation/'+cropTitle[cp]+'/'+str(var[iplot])+'_'+cropTitle[cp]+'_yield_corr',dpi=500)
    show() 
    clf()

    
           
###############################################
# Plot Slopes
###############################################
cp=2

cmapArray=plt.cm.gist_rainbow(arange(256))
y1=0
y2=255

MinMaxArray=ones(shape=(2,3))

b=0
significant=0
highlySignificant=0
countyCounter=0

MinMaxArray=zeros(shape=(2,1))
goodCity2=zeros(shape=(3143,3),dtype=bool)

for iplot in range(3,5):
    print iplot
    
    figure(5,figsize=(9,7))
    show()
    
    for icity in range(3143):
        if slope[iplot,icity,cp]!=0 or slope[iplot,icity,cp]!=-9999:
            goodCity2[icity,cp]=True
    
    cmax=np.amax(slope[iplot,highTotalYield[:,cp],cp])
    #cmax=0
    cmin=np.amin(slope[iplot,highTotalYield[:,cp],cp])
    if cp==0:
        if iplot==2:
            cmin=-0.1*10
            cmax=0.02*10
        if iplot==3:
            cmin=-8
            cmax=1.25
        if iplot==4:
            cmin=-4
            cmax=.7
            
    if cp==1:
        if iplot==2:
            cmin=-0.02*10
            cmax=0.006*10
        if iplot==3:
            cmin=-1.3
            cmax=.3
        if iplot==4:
            cmin=-.8
            cmax=.2
            
    if cp==2:
        if iplot==3:
            cmax=20
            cmin=-85
        if iplot==4:
            cmax=15
            cmin=-50
            
    if cmin==-9999:
        print 'cmin=-9999'
    
    j=-1
    
    subPlot1 = plt.axes([.5,.25,.475,.5])
    MinMaxArray[0,0]=cmax
    MinMaxArray[1,0]=cmin
    
    plt.imshow(MinMaxArray,cmap='gist_rainbow')
    plt.colorbar()
    
    # create the map
    subPlot1 = plt.axes([0.1,0,0.8,1])
    m = Basemap(llcrnrlon=-107,llcrnrlat=27,urcrnrlon=-71,urcrnrlat=49,
    lat_ts=50,resolution='i',area_thresh=10000)
            
    # load the shapefile, use the name 'states'
    m.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
    
    ax = plt.gca() # get current axes instance
    
    for shape_dict in m.states_info:
        j+=1
        seg = m.states[j]
        
        s=int(shape_dict['STATEFP'])
        c=int(shape_dict['COUNTYFP'])
        
        if s==72 or cIndex[s,c]==-9999:
            continue
            
        if highTotalYield[cIndex[s,c],cp]==False:
            continue
            
        x=slope[iplot,cIndex[s,c],cp]
        if math.isnan(x)==True:
            b+=1
            goodCity2[cIndex[s,c],cp]=False
            continue

        y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
        icmap1=min(255,int(round(y,1)))
        icmap=max(0,icmap1)
        
         
        poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
        ax.add_patch(poly)
    title('Slope: '+cropTitle[cp]+' Yield/'+titleVar[iplot]+', Avg = '+str(round(mean(slope[iplot,highTotalYield[:,cp],cp]),2)))
    savefig('final_figures/slope/'+cropTitle[cp]+'/'+str(var[iplot])+'_'+cropTitle[cp]+'_yield_slope',dpi=500)
    show() 
    pause(1)
    clf()
    
    
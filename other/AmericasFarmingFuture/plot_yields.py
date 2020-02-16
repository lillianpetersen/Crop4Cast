###########################################################
# Plots future crop yields
###########################################################
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
    
    

futureYield=pickle.load(open('pickle_files/futureYield.p','rb'))
# futureYield dimensions = (nCities,nPredictor,nyears,nScen,nCrop)
# nPredictor  0=Summer avg, 1=Heat Waves, 2=KDD
cropYield=pickle.load(open('pickle_files/cropYield.p','rb'))
# cropYield dimensions = (nCities,year,cp)
cIndex=pickle.load(open('pickle_files/cIndex.p','rb'))
presentGrowingCounties=pickle.load(open('pickle_files/presentGrowingCounties.p','rb'))

nyears=32
cp=0

icity=789
avgFutureYield=zeros(shape=(nyears))
avgCropYield=zeros(shape=(116))        

for y in range(nyears):
    avgFutureYield[y]=mean(futureYield[presentGrowingCounties[cp],0:2,y,0,cp])
for y in range(70,116):
    avgCropYield[y]=mean(cropYield[presentGrowingCounties[cp],y,cp])


x=range(2020,2052)
ydata=avgFutureYield 
#ydataAvg=Avg(ydata)
#slope,b=polyfit(x,ydata,1)
#yfit=slope*x+b

x2=range(1970,2016)
ydata2=avgCropYield[70:116]
#ydataAvg2=Avg(ydata2)
#slope2,b2=polyfit(x2,ydata2,1)
#yfit2=slope2*x2+b2

#plot(x,ydata,'*b',x,yfit,'g',x2,ydata2,'*k',x2,yfit2,'g')
plot(x2,ydata2,'-*g',x,ydata,'-*b')
legend(['past yield','future yield'],loc='upper left')
ylabel('Yield')
xlabel('year')
title('Past vs Future Yield')
grid(True)
show()
clf()

# create the map
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# load the shapefile, use the name 'states'
map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)

ax = plt.gca() # get current axes instance
j=0
cmapArray=plt.cm.jet(arange(256))
cmin=0
y1=0
y2=256
year=100
cp=2
cmax=np.amax(cropYield[:,year,cp])

for shape_dict in map.states_info:
    seg = map.states[j]
    
    s=int(shape_dict['STATEFP'])
    c=int(shape_dict['COUNTYFP'])
    
    if s==72 or cIndex[s,c]==-9999:
        j+=1
        continue

    Yield=cropYield[cIndex[s,c],year,cp]
    
    if Yield==-9999.0 or Yield==0:
        j+=1
        continue
        
    x=Yield
    y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
    icmap=min(255,int(round(y,1)))
       
    poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
    ax.add_patch(poly)
       
    j+=1
title('Crop '+str(cp)+' Yield for Year '+str(year+1900))
plt.show() 

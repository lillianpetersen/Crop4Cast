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
    
    
### Running mean/Moving average
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
       
###############################################
# Variables
###############################################
futureYield=pickle.load(open('pickle_files/futureYield.p','rb'))
# futureYield dimensions = (nCities,nPredictor,nyears,nScen,nCrop)
# nPredictor  0=KDD, 1=Summer Avg, 2=Heat Waves
cropYield=pickle.load(open('pickle_files/cropYield.p','rb'))
# cropYield dimensions = (nCities,year,cp)
NormalizedCropYield=pickle.load(open('pickle_files/NormalizedCropYield.p','rb'))
cIndex=pickle.load(open('pickle_files/cIndex.p','rb'))
cIndexState=pickle.load(open('pickle_files/cIndexState.p','rb'))
presentGrowingCounties=pickle.load(open('pickle_files/presentGrowingCounties.p','rb'))
SummerAvg=pickle.load(open('pickle_files/SummerAvg.p','rb'))
HeatWaves=pickle.load(open('pickle_files/HeatWaves.p','rb'))
KDDays=pickle.load(open('pickle_files/KDD.p','rb'))
highTotalYield=pickle.load(open('pickle_files/highTotalYield.p','rb'))
dataAt2015=pickle.load(open('pickle_files/dataAt2015.p','rb'))
slope=pickle.load(open('pickle_files/slope.p','rb'))
bIntercept=pickle.load(open('pickle_files/bIntercept.p','rb'))
SummerAvgFuture=pickle.load(open('pickle_files/SummerAvgFuture.p','rb'))
BBcounty2=pickle.load(open('pickle_files/BBcounty2.p','rb'))

nScen=2
nCrop=3
goodfuture=zeros(shape=(nScen,3143,nCrop),dtype=bool)
goodpast=zeros(shape=(3143,nCrop),dtype=bool)
goodpastyears=zeros(shape=(3143,116,nCrop),dtype=bool)
nyears=100
avgFutureYield=zeros(shape=(nScen,nyears,nCrop))
avgFutureYieldIndividualCities=zeros(shape=(nScen,3143,nyears,nCrop))
avgCropYield=zeros(shape=(116,nCrop))
futureYieldPlusTrend=zeros(shape=(nScen,3143,100,nCrop))
avgFutureYieldPlusTrend=zeros(shape=(nScen,100,nCrop))
avgNormalizedCropYield=zeros(shape=(116,nCrop))  
SummerAvgAvg=zeros(shape=(116,nCrop)) 
goodSummer=zeros(shape=(3143,nCrop),dtype=bool)
goodSummeryears=zeros(shape=(3143,116,nCrop),dtype=bool)

SummerAvgAvg2=zeros(shape=(116)) 
goodSummer2=zeros(shape=(3143),dtype=bool)
goodSummeryears2=zeros(shape=(3143,116),dtype=bool)

SummerAvgAvgFuture=zeros(shape=(2,116)) 
goodSummerFuture=zeros(shape=(2,3143),dtype=bool)
goodSummeryearsFuture=zeros(shape=(2,3143,116),dtype=bool)

cropTitle=('Corn','Soybean','Rice') 
subtract=(20,5,200)
units=('Bushels/Acre','Bushels/Acre','Pounds/Acre')

for scen in range(nScen):
    for y in range(16,100):
        for cp in range(nCrop):
            for icity in range(3143):
                if presentGrowingCounties[icity,cp]==False:
                    continue
                avgFutureYieldIndividualCities[scen,icity,y,cp]=mean(futureYield[icity,0:2,y,scen,cp])
            
yr=arange(2016,2100)
x=yr
for icity in range(3143): 
    if sum(presentGrowingCounties[icity,:])==0 or icity==24:
        continue           
    for cp in range(nCrop):
        if presentGrowingCounties[icity,cp]==False:
            continue
        for scen in range(nScen):
            futureYieldPlusTrend[scen,icity,16:100,cp]=avgFutureYieldIndividualCities[scen,icity,16:100,cp]-dataAt2015[icity,cp]
            futureYieldPlusTrend[scen,icity,16:100,cp]=futureYieldPlusTrend[scen,icity,16:100,cp]+(slope[0,icity,cp]*x+bIntercept[0,icity,cp])+subtract[cp]
        
        
for icity in range(3143):
    for scen in range(nScen):
        for y in range(16,100):
            for cp in range(nCrop):
                if futureYield[icity,0,y,scen,cp]>0 and presentGrowingCounties[icity,cp]==1 and isnan(futureYield[icity,0,y,scen,cp])==False:
                    goodfuture[scen,icity,cp]=True
            if SummerAvgFuture[scen,icity,y]>0 and BBcounty2[icity]==True:
                    goodSummeryearsFuture[scen,icity,y]=True

    for y in range(70,116):
        for cp in range(nCrop):
            if cropYield[icity,y,cp]>0 and highTotalYield[icity,cp]==True:
                    goodpast[icity,cp]=True
                    goodpastyears[icity,y,cp]=True
            if SummerAvg[icity,y]>0 and highTotalYield[icity,cp]==True:
                    goodSummeryears[icity,y,cp]=True
            if SummerAvg[icity,y]>0 and BBcounty2[icity]==True:
                    goodSummeryears2[icity,y]=True
                            
for scen in range(nScen):
    for y in range(16,100):
        for cp in range(nCrop):
            avgFutureYield[scen,y,cp]=mean(futureYield[goodfuture[scen,:,cp],0:2,y,scen,cp])+subtract[cp]
            avgFutureYieldPlusTrend[scen,y,cp]=mean(futureYieldPlusTrend[scen,goodfuture[scen,:,cp],y,cp])
    
j=zeros(shape=(116,nCrop))
summerj=zeros(shape=(116,nCrop))
summerjFuture=zeros(shape=(2,116))
summerj2=zeros(shape=(116))
for cp in range(nCrop):
    for icity in range(3143):
        if goodpast[icity,cp]==False:
            continue
        for y in range(70,116):
            if goodpastyears[icity,y,cp]==True:
                avgCropYield[y,cp]+=cropYield[icity,y,cp]
                avgNormalizedCropYield[y,cp]+=NormalizedCropYield[icity,y,cp]
                j[y,cp]+=1
            if goodSummeryears[icity,y,cp]==True:
                SummerAvgAvg[y,cp]+=SummerAvg[icity,y]
                summerj[y,cp]+=1
                
            if goodSummeryears2[icity,y]==True:
                SummerAvgAvg2[y]+=SummerAvg[icity,y]
                summerj2[y]+=1 
                
        for y in range(16,100):
            for scen in range(2):
                if goodSummeryearsFuture[scen,icity,y]==False:
                    continue
                SummerAvgAvgFuture[scen,y]+=SummerAvgFuture[scen,icity,y]
                summerjFuture[scen,y]+=1                

for y in range(70,116):
    for cp in range(nCrop):
        avgCropYield[y,cp]=avgCropYield[y,cp]/j[y,cp]
        SummerAvgAvg[y,cp]=SummerAvgAvg[y,cp]/summerj[y,cp]
        avgNormalizedCropYield[y,cp]=(avgNormalizedCropYield[y,cp]/j[y,cp])#-subtract[cp]
    SummerAvgAvg2[y]=SummerAvgAvg2[y]/summerj2[y]
for y in range(16,100):
    for scen in range(2):
        SummerAvgAvgFuture[scen,y]=SummerAvgAvgFuture[scen,y]/summerjFuture[scen,y]
'''
###############################################
# Future and Past Summer Avg temp
###############################################
x=range(2016,2100)
ydata=SummerAvgAvgFuture[0,16:100]
yfit=movingaverage(ydata,5)

x2=range(1970,2016)
ydata2=SummerAvgAvg2[70:116]
yfit2=movingaverage(ydata2,5)

x3=range(2016,2100)
ydata3=SummerAvgAvgFuture[1,16:100]
yfit3=movingaverage(ydata3,5)

#plot(x,ydata,'*b',x,yfit,'g',x2,ydata2,'*k',x2,yfit2,'g')
plot(x2[2:44],yfit2[2:44],'-g',linewidth=3)
plot(x3[2:82],yfit3[2:82],'-b',linewidth=3)
plot(x[2:82],yfit[2:82],'-r',linewidth=3)
plot(x2,ydata2,'-g',linewidth=1)
plot(x,ydata,'-r',linewidth=1)
plot(x3,ydata3,'-b',linewidth=1)
legend(['past: historical','future: low emissions','future: high emissions'],loc='lower center')
ylabel(cropTitle[cp]+' Yield, '+units[cp])
xlabel('year')
title('Bread Basket Summer Average Temp: Past vs Future')
grid(True)
savefig('final_figures/poster_final/past_vs_future_summer',dpi=700)
show()

clf()    
    

###############################################
# Past Yields
###############################################

for cp in range(nCrop):
    
    x2=np.arange(1970,2016)
    ydata2=avgCropYield[70:116,cp]
    ydataAvg2=Avg(ydata2)
    slope2,b2=polyfit(x2,ydata2,1)
    yfit2=slope2*x2+b2
    
    plot(x2,ydata2,'--*b',x2,yfit2,'-g')
    ylabel(cropTitle[cp]+' Yield, Bushels/Acre')
    xlabel('year')
    title(cropTitle[cp]+' Yield: Avg of Corn Growing Counties, slope='+str(round(slope2,2))+' Bu/Acre/Year')
    grid(True)
    savefig('final_figures/yield/'+cropTitle[cp]+'/past_yield_'+cropTitle[cp],dpi=700)
    show()
    exit()
    clf()
  
###############################################
# Summer Avg correlation with Yield
###############################################
ticks=([110,120,130,140,150,160,170,180,190,200,210,220],[35,40,45,50,55,60])

for cp in range(nCrop):
    if cp==2:
        continue
    x=np.arange(1970,2016)
    ydata=SummerAvgAvg[70:116,cp]
    #ydataAvg=Avg(ydata)
    #slope,b=polyfit(x,ydata,1)
    #yfit=slope*x+b
    
    x2=range(1970,2016)
    ydata2=avgNormalizedCropYield[70:116,cp]
    #ydataAvg2=Avg(ydata2)
    #slope2,b2=polyfit(x2,ydata2,1)
    #yfit2=slope2*x2+b2
    
    fig, ax1 = plt.subplots()
    
    ax2 = ax1.twinx()
    ax1.plot(x,ydata,'-*g')
    ax2.plot(x2,ydata2,'-*b')
    ax1.set_yticks([70,75,80,85,90])
    ax2.set_yticks(ticks[cp])
    ax2.set_ylabel(cropTitle[cp]+' Yield',color='b')
    ax1.set_ylabel('Summer Avg Temperature',color='g')
    ax1.set_xlabel('year')
    title(cropTitle[cp]+' Yield and Summer Avg Temperature')
    savefig('final_figures/yield/'+cropTitle[cp]+'/'+cropTitle[cp]+'yield_and_summer_avg_temp',dpi=700)
    show()
    clf()    

'''
###############################################
# Future and Past Yields
###############################################
for cp in range(3):
    #figure(1,figsize=(9,5))
    x=np.arange(2016,2100)
    ydata=avgFutureYield[0,16:100,cp]
    yfit=movingaverage(ydata,5)
    slope,bIntercept=polyfit(x,ydata,1)
    yfitline=slope*x+bIntercept
    
    x2=np.arange(1970,2016)
    ydata2=avgCropYield[70:116,cp]
    yfit2=movingaverage(ydata2,5)
    slope2,bIntercept2=polyfit(x2,ydata2,1)
    yfitline2=slope2*x2+bIntercept2
    
    x3=np.arange(2016,2100)
    ydata3=avgFutureYield[1,16:100,cp]
    yfit3=movingaverage(ydata3,5)
    slope3,bIntercept3=polyfit(x3,ydata3,1)
    yfitline3=slope3*x3+bIntercept3
    
    #plot(x,ydata,'*b',x,yfit,'g',x2,ydata2,'*k',x2,yfit2,'g')
    plot(x2[2:44],yfit2[2:44],'-g',linewidth=3)
    plot(x3[2:82],yfit3[2:82],'-b',linewidth=3)
    plot(x[2:82],yfit[2:82],'-r',linewidth=3)
    
    plot(x2,ydata2,'-g',linewidth=1)
    plot(x,ydata,'-r',linewidth=1)
    plot(x3,ydata3,'-b',linewidth=1)
    
    #plot(x2,yfitline2, '--g',linewidth=2)
    #plot(x,yfitline, '--r',linewidth=2)
    #plot(x3,yfitline3, '--b',linewidth=2)
    legend(['past: historical','future: low emissions','future: high emissions'],loc='lower center')
    ylabel(cropTitle[cp]+' Yield, '+units[cp])
    xlabel('year')
    title('U.S. '+cropTitle[cp]+' Yield: Past vs Future')
    grid(True)
    #savefig('final_figures/yield/'+cropTitle[cp]+'/past_vs_future_yield_'+cropTitle[cp]+'_little',dpi=700)
    show()
    print cropTitle[cp],'\nhistorical:',round((slope2*10/yfitline2[0])*100,2),'\nfuture high:',round((slope*10/yfitline[0])*100,2), '\nfuture low:',round((slope3*10/yfitline3[0])*100,2)
    
    pause(3)
    clf() 
exit()
###############################################
# Future and Normalized Past Yields
###############################################
for cp in range(nCrop):
    
    x=range(2016,2100)
    ydata=avgFutureYield[0,16:100,cp]
    yfit=movingaverage(ydata,5)
    
    x2=range(1970,2016)
    ydata2=avgNormalizedCropYield[70:116,cp]
    yfit2=movingaverage(ydata2,5)
    
    x3=range(2016,2100)
    ydata3=avgFutureYield[1,16:100,cp]
    yfit3=movingaverage(ydata3,5)
    
    #plot(x,ydata,'*b',x,yfit,'g',x2,ydata2,'*k',x2,yfit2,'g')
    plot(x2[2:44],yfit2[2:44],'-g',linewidth=3)
    plot(x3[2:82],yfit3[2:82],'-b',linewidth=3)
    plot(x[2:82],yfit[2:82],'-r',linewidth=3)
    plot(x2,ydata2,'-g',linewidth=1)
    plot(x,ydata,'-r',linewidth=1)
    plot(x3,ydata3,'-b',linewidth=1)
    legend(['past: detrended','future: low emissions','future: high emissions'],loc='lower center')
 
    ylabel(cropTitle[cp]+' Yield, '+units[cp])
    xlabel('year')
    title('U.S. '+cropTitle[cp]+' Yield: Past vs Future')
    grid(True)
    savefig('final_figures/yield/'+cropTitle[cp]+'/normalized_past_vs_future_yield_'+cropTitle[cp],dpi=700)
    show()
    pause(5)
    clf()    

###############################################
# Future and Past Yields with added in trend
###############################################

for cp in range(nCrop):
    
    x=range(2016,2100)
    ydata=avgFutureYieldPlusTrend[0,16:100,cp]
    yfit=movingaverage(ydata,5)
    
    
    x4=np.arange(2018,2100)
    ydata2=avgCropYield[70:116,cp]
    yfit2=movingaverage(ydata2,5)
    x2=np.arange(1970,2016)
    slope,bIntercept=polyfit(x2,ydata2,1)
    yfit21=slope*x4+bIntercept
    
    x3=range(2016,2100)
    ydata3=avgFutureYieldPlusTrend[1,16:100,cp]
    yfit3=movingaverage(ydata3,5)
    
    #plot(x,ydata,'*b',x,yfit,'g',x2,ydata2,'*k',x2,yfit2,'g')
    plot(x2[2:44],yfit2[2:44],'-g',linewidth=3)

    plot(x3[2:82],yfit3[2:82],'-b',linewidth=3)
    plot(x[2:82],yfit[2:82],'-r',linewidth=3)
    plot(x2,ydata2,'-g',linewidth=1)
    plot(x,ydata,'-r',linewidth=1)
    plot(x3,ydata3,'-b',linewidth=1)
    plot(x4,yfit21,'--g',linewidth=3)
    legend(['past: historical','future: low emissions with trend','future: high emissions with trend'],loc='lower right')
    ylabel(cropTitle[cp]+' Yield, '+units[cp])
    xlabel('year')
    title('U.S. '+cropTitle[cp]+' Yield: Past vs Future')
    grid(True)
    savefig('final_figures/yield/'+cropTitle[cp]+'/past_vs_future_yield_plus_trend_'+cropTitle[cp]+'_little',dpi=700)
    show()
    pause(3)
    clf()

###############################################
# Past Yields US map
###############################################
for cp in range(nCrop):
    #cmapArray=concatenate((plt.cm.jet(arange(128)),plt.cm.PuOr(arange(128))),axis=0)
    j=-1
    MinMaxArray=ones(shape=(2,1))
    
    cmapArray=plt.cm.jet(arange(256))
    cmin=0
    y1=0
    y2=256
    cmax=.8*(np.amax(cropYield[:,105:116,cp]))
    
    subPlot1 = plt.axes([.5,.2,.475,.6])
    MinMaxArray[0,0]=cmin
    MinMaxArray[1,0]=cmax
    plt.imshow(MinMaxArray,cmap='jet')
    plt.colorbar()
    
    # create the map
    subPlot1 = plt.axes([0.1,0,0.8,1])
    map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    
    # load the shapefile, use the name 'states'
    map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
    
    ax = plt.gca() # get current axes instance
    
    for shape_dict in map.states_info:
        j+=1
        seg = map.states[j]
        
        s=int(shape_dict['STATEFP'])
        c=int(shape_dict['COUNTYFP'])
        
        if s==72 or cIndex[s,c]==-9999:
            continue
    
        Yield=mean(cropYield[cIndex[s,c],70:81,cp])
        
        if Yield<=0:
            continue
            
        x=Yield
        y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
        icmap=min(255,int(round(y,1)))
        
        poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
        ax.add_patch(poly)
    
    title('Avg '+units[cp]+' of '+cropTitle[cp]+' by County 1970-1980')       
    savefig('final_figures/yield/'+cropTitle[cp]+'/'+'1970-1980_yield_us_map_'+cropTitle[cp],dpi=700)
    show()
    clf()

###############################################
# Present Yields US map
###############################################
for cp in range(nCrop):
    j=-1
    MinMaxArray=ones(shape=(2,1))
    
    cmapArray=plt.cm.jet(arange(256))
    cmin=0
    y1=0
    y2=256
    cmax=.8*(np.amax(cropYield[:,105:116,cp]))
    
    subPlot1 = plt.axes([.5,.2,.475,.6])
    MinMaxArray[0,0]=cmin
    MinMaxArray[1,0]=cmax
    plt.imshow(MinMaxArray,cmap='jet')
    plt.colorbar()
    
    # create the map
    subPlot1 = plt.axes([0.1,0,0.8,1])
    map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    
    # load the shapefile, use the name 'states'
    map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
    
    ax = plt.gca() # get current axes instance
    
    for shape_dict in map.states_info:
        j+=1
        seg = map.states[j]
        
        s=int(shape_dict['STATEFP'])
        c=int(shape_dict['COUNTYFP'])
        
        if s==72 or cIndex[s,c]==-9999:
            continue
        
        Yield=mean(cropYield[cIndex[s,c],105:116,cp])
        
        if Yield<=0:
            continue
            
        x=Yield
        y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
        icmap=min(255,int(round(y,1)))
        
        poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
        ax.add_patch(poly)
    
    title('Avg '+units[cp]+' of '+cropTitle[cp]+' by County 2005-2015')       
    savefig('final_figures/yield/'+cropTitle[cp]+'/'+'2005-2015_yield_us_map_'+cropTitle[cp],dpi=700)
    show() 
    clf()
 
###############################################
# Future Yields US map
###############################################
scen=0
for cp in range(nCrop):
    j=-1 #counter for how many times loop goes through
    
    MinMaxArray=ones(shape=(2,1))
    cmapArray=plt.cm.jet(arange(256))
    cmin=0
    y1=0
    y2=256
    cmax=.8*(np.amax(cropYield[:,105:116,cp]))
    
    subPlot1 = plt.axes([.5,.2,.475,.6])
    MinMaxArray[0,0]=cmin
    MinMaxArray[1,0]=cmax
    plt.imshow(MinMaxArray,cmap='jet')
    plt.colorbar()
    
    # create the map
    subPlot1 = plt.axes([0.1,0,0.8,1])
    map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    
    # load the shapefile, use the name 'states'
    map.readshapefile('data/shape_files/counties/cb_2015_us_county_20m', name='states', drawbounds=True)
    
    ax = plt.gca() # get current axes instance
    
    for shape_dict in map.states_info:
        j+=1
        seg = map.states[j]
        
        s=int(shape_dict['STATEFP'])
        c=int(shape_dict['COUNTYFP'])
    
        if s==72 or cIndex[s,c]==-9999:
            continue
        
        Yield=mean(futureYield[cIndex[s,c],:,90:100,scen,cp])    
        
        if isnan(Yield):
            continue
        
        if Yield<=0:
            continue
            
        x=Yield
        y=y1+(y2-y1)/(cmax-cmin)*(x-cmin)
        icmap=min(255,int(round(y,1)))
        
        poly = Polygon(seg,facecolor=[cmapArray[icmap,0],cmapArray[icmap,1],cmapArray[icmap,2]],edgecolor=[0,0,0])
        ax.add_patch(poly)
    
    title('Avg '+units[cp]+' of '+cropTitle[cp]+' by County 2090-2100, High Emissions')       
    savefig('final_figures/yield/'+cropTitle[cp]+'/'+'2090-2100_yield_us_map_'+cropTitle[cp],dpi=700)
    show() 
    clf() 
        
    
    
    
#presentGrowingCounties=zeros(shape=(3143,3))
#for icity in range(3143):
#    for cp in range(3):
#        Yield=mean(cropYield[icity,105:115,cp])
#        if Yield>0 or highTotalYield[icity,cp]==True:
#            presentGrowingCounties[icity,cp]=1

    
       

    
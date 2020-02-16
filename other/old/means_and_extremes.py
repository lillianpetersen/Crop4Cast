###############################################
# The Main Code
# Reads in data
# gets rid of bad cities
# computes the means and extremes
# plots and records slope, corr, lat, lon, station name
###############################################
from pylab import *
import csv
from math import sqrt
from sys import exit
import numpy as np
import pickle
import os
from scipy import stats
import matplotlib.lines as lines

# cd Documents/Science_Fair_2017_Crop_Yields/
wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'

# choose what you want the code to run for #
useUniformStartDate=False
dataYears=10
goodCityRatio=.9
makePlots=True
nCrop=1
country='ethiopia'

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
    
def stdDev(x):   
    '''function to compute standard deviation'''
    xAvg=np.mean(x)
    xOut=0.
    for k in range(len(x)):
        xOut=xOut+(x[k]-xAvg)**2
    xOut=xOut/(k+1)
    xOut=sqrt(xOut)
    return xOut
    
def Variance(x):   
    '''function to compute the variance (std dev squared)'''
    xAvg=np.mean(x)
    xOut=0.
    for k in range(len(x)):
        xOut=xOut+(x[k]-xAvg)**2
    xOut=xOut/(k+1)
    return xOut
    
def SumOfSquares(x):
    '''function to compute the sum of squares'''
    xOut=0.
    for k in range(len(x)):
        xOut=xOut+x[k]**2
    return xOut

def corr(x,y):   
    ''' function to find the correlation of two arrays'''
    xAvg=np.mean(x)
    Avgy=np.mean(y)
    rxy=0.
    n=min(len(x),len(y))
    for k in range(n):
        rxy=rxy+(x[k]-xAvg)*(y[k]-Avgy)
    rxy=rxy/(k+1)
    stdDevx=stdDev(x)
    stdDevy=stdDev(y)
    rxy=rxy/(stdDevx*stdDevy)
    return rxy
    
###############################################
# Read in Weather Data
###############################################
cityFile='tmax_cities_over_'+str(dataYears)+'_years' # what file to read in #

nCities=3143 # number of cities   
#f=open(wd+'written_files/'+cityFile+'.txt','r')
f=open(wd+'written_files/tmax_nearest_stations.txt')

# read in the pickle files
#startGrowingMon=pickle.load(open('pickle_files/startGrowingMon.p','rb'))
#startGrowingDay=pickle.load(open('pickle_files/startGrowingDay.p','rb'))
#endGrowingMon=pickle.load(open('pickle_files/endGrowingMon.p','rb'))
#endGrowingDay=pickle.load(open('pickle_files/endGrowingDay.p','rb'))
#cIndex=pickle.load(open('pickle_files/cIndex.p','rb'))
#cIndexState=pickle.load(open('pickle_files/cIndexState.p','rb'))
#cropYield=pickle.load(open('pickle_files/cropYield.p','rb'))
#countyName=pickle.load(open('pickle_files/countyName.p','rb'))
#presentGrowingCounties=pickle.load(open('pickle_files/presentGrowingCounties.p','rb'))

production=np.load(wd+'saved_vars/ethiopia/production.npy')
cropYield=np.load(wd+'saved_vars/ethiopia/cropYield.npy')
areaHarv=np.load(wd+'saved_vars/ethiopia/areaHarvested.npy')

production=production[:,72]
cropYield=cropYield[:,72]
areaHarv=areaHarv[:,72]

# initialize variables
icity=-1 # a counter of the number of cities
slope=-9999.*ones(shape=(5,nCities,nCrop))
bIntercept=-9999.*ones(shape=(5,nCities,nCrop))
lat=-9999.*ones(shape=(nCities))
lon=-9999.*ones(shape=(nCities))
lat2=-9999.*ones(shape=(nCities))
lon2=-9999.*ones(shape=(nCities))
stationList=[]
cityList=[]
goodCity=ones(shape=(4,nCities),dtype=bool)
Corr=np.zeros(shape=(5,nCities,nCrop))
R2=-9999.*np.ones(shape=(5,nCities,nCrop))
maxCorr=np.zeros(shape=(nCities,nCrop))
iplotCorr=np.zeros(shape=(nCities,nCrop))
maxCorr2=np.zeros(shape=(nCities,nCrop))
iplotCorr2=np.zeros(shape=(nCities,nCrop))
T90all=np.zeros(shape=(nCities,2))
T10all=np.zeros(shape=(nCities,2))
runCrops=np.zeros(shape=(3))
dataAt2015=-9999*np.ones(shape=(nCities,nCrop))

HeatWaves=-9999*np.ones(shape=(3143,116))
SummerAvg=-9999*np.ones(shape=(3143,116))
KDDays=-9999*np.ones(shape=(3143,116))
mostcity=[600,604,632,644,647,651,843]

for cityline in f:  # read in the file with the closest counties
    icity+=1
    
    #if sum(presentGrowingCounties[icity])==0:
    #    continue
    
    
    #if icity!=604:
    #    continue
        
    doCorn=1
    doSoy=1
    doRice=1
        
    # initialize more variables
    cityline=cityline.translate(None, ' ')
    tmp=cityline.split(',')
    station=tmp[0]
    stationList.append(station)
    lat[icity]=tmp[2]
    lon[icity]=tmp[3]
    iBeg1=float(tmp[5])-1900
    iBeg1=int(iBeg1)
    if useUniformStartDate:
        iBeg=(2015-dataYears)-1900
    iEnd1=float(tmp[6])-1900
    iEnd1=int(iEnd1)
    #iEnd=float(tmp[5])-1900
    #iEnd=int(iEnd)
    if iEnd1>116:
        iEnd1=116
    city=tmp[7].strip().title()
    #city=countyName[icity].title()
     
    station2=tmp[8]
    lat2[icity]=tmp[10]
    lon2[icity]=tmp[11]
    iBeg2=float(tmp[13])-1900
    iBeg2=int(iBeg2)
    iEnd2=float(tmp[14])-1900
    iEnd2=int(iEnd2)

    iEnd=max(iEnd1,iEnd2)
    iBeg=min(iBeg1,iBeg2)
    
    print '\n'
    print icity,'of',nCities
    print city.title()
   
    # initialize more variables
    #dailyData1=-9999.*ones(shape=(4,116,12,31))
    dailyData1=-9999.*ones(shape=(2,116,12,31))
    badData=zeros(shape=(2,116))  
    badYears=ones(shape=(2,116))
    badYearsforAvg=ones(shape=(2,116,nCrop))
    badYearYield=ones(shape=(2,116,nCrop))
    badYearYieldBoolean=zeros(shape=(2,116,nCrop),dtype=bool)
    badYear=ones(shape=(2,116))
        
    dailyData2=-9999.*ones(shape=(2,116,12,31))
    badData2=zeros(shape=(2,116))  
        
    f = open(wddata+'noaa_daily_data/ghcnd_all/'+station+'.dly', 'r')
    f2=open(wddata+'noaa_daily_data/ghcnd_all/'+station2+'.dly', 'r')
    
    for line in f: #read in the daily data for each closest station
        # initialize tmp variables
        tmp=line[0:270]
        station=tmp[0:11]
        year=float(tmp[11:15])
        month=float(tmp[15:17])
        var=tmp[17:21]
        if var=='TMAX':
            v=0
        elif var=='TMIN':
            v=1
        else:
            v=-1
        
        ife:v!=-1:   
            m=int(month-1)
            y=int(year-1900)
            d=21
            if y<0:
                y=0
            for d in range(31):
                tmp2=tmp[21+8*d:21+8*d+5]
                if tmp2=='   0T' or tmp2=='   0P':
                    tmp2='    0'
                if int(tmp2)!=-9999:
                    #if v==0 or v==1:
                    tmp3=(9./5.*float(tmp2)/10.+32.)
                    #else:
                    #    tmp3=(float(tmp2)/10/10/2.54)
                else:
                    badData[v,y]+=1
                    tmp3=-9999
                dailyData1[v,y,m,d]=tmp3 # put the data into one 4D array
    #b=0
    #g=0
    #for y in range(iBeg,iEnd):
    #    for m in range(12):
    #        for d in range(31):
    #            if dailyData[0,y,m,d]<-100:
    #                b+=1
    #            else:
    #                g+=1
    
    for line in f2: #read in the daily data for each second closest station
        # initialize tmp variables
        tmp=line[0:270]
        station=tmp[0:11]
        year=float(tmp[11:15])
        month=float(tmp[15:17])
        var=tmp[17:21]
        if var=='TMAX':
            v=0
        elif var=='TMIN':
            v=1
        #elif var=='PRCP':
        #    v=2
        #elif var=='SNOW':
        #    v=3
        else:
            v=-1
        
        if v!=-1:   
            m=int(month-1)
            y=int(year-1900)
            d=21
            if y<0:
                y=0
            for d in range(31):
                tmp2=tmp[21+8*d:21+8*d+5]
                if tmp2=='   0T' or tmp2=='   0P':
                    tmp2='    0'
                if tmp2!='-9999':
                    #if v==0 or v==1:
                    tmp3=(9./5.*float(tmp2)/10.+32.)
                    #else:
                    #    tmp3=(float(tmp2)/10/10/2.54)
                else:
                    badData2[v,y]+=1
                    tmp3=-9999
                dailyData2[v,y,m,d]=tmp3 # put the data into one 4D array
    
    #################################################
    # Average the closest and second closest station
    #################################################
    dailyData=-9999*ones(shape=(2,116,12,31))
    mAvg=np.zeros(shape=(2,116,12))  #monthly average
    goodMonths=np.zeros(shape=(2,116,12))
    #goodydcorn=0
    #goodydsoy=0
    #goodydrice=0
    
    for v in range(2):
        for y in range(iBeg,iEnd):
            g=0 # number of good months
            cp=0
            for m in range(12):
                j=0    # j is number of good days in month
                b=0    # b is the number of bad days in a month
                for d in range(31):
                    if dailyData1[v,y,m,d]>-100 and dailyData2[v,y,m,d]>-100 and dailyData1[v,y,m,d]!=0.00 and dailyData2[v,y,m,d]!=0.00: 
                        j+=1 
                        dailyData[v,y,m,d]=(dailyData1[v,y,m,d]+dailyData2[v,y,m,d])/2
                        
                    elif dailyData1[v,y,m,d]>-100 and dailyData1[v,y,m,d]!=0.00 and dailyData2[v,y,m,d]<-100:
                        j+=1
                        dailyData[v,y,m,d]=dailyData1[v,y,m,d]
                        
                    elif dailyData1[v,y,m,d]<-100 and dailyData2[v,y,m,d]>-100 and dailyData2[v,y,m,d]!=0.00:
                        j+=1
                        dailyData[v,y,m,d]=dailyData2[v,y,m,d]
                        
                    else: # dailyData1[v,y,m,d]<-100 and dailyData2[v,y,m,d]<-100 and d!=30:
                        b+=1
                        dailyData[v,y,m,d]=dailyData[v,y,m,d-1]
                if j>25:
                    g+=1
                        
                if j>=15:
                    goodMonths[v,y,m]=1 # one means the month is good
                if j<15:   # make sure the month has atleast 20 good day
                    goodMonths[v,y,m]=0
                    mAvg[v,y,m]=-9999
            if goodMonths[v,y,3]==1 and goodMonths[v,y,4]==1 and goodMonths[v,y,5]==1 and goodMonths[v,y,6]==1 and goodMonths[v,y,7]==1 and goodMonths[v,y,8]==1 and goodMonths[v,y,9]==1 and goodMonths[v,y,10]==1:
                badYear[v,y]=0 # zero means the year is good

            if sum(goodMonths[v,y])==12:
                badYears[v,y]=0 # zero means the year is good
                
        if sum(badYear[v,iBeg:iEnd])/float(dataYears) > goodCityRatio: # if the city has too much bad data
            goodCity[v,icity]=False # True means the city is good
        
        #for cp in range(nCrop):        
        for y in range(iBeg,iEnd):
            if cropYield[y]>0 and badYear[v,y]==0:
                cp=0
                badYearYield[v,y,cp]=0 # zero means the year is good
                badYearYieldBoolean[v,y,cp]=True
            if cropYield[y]!=-9999 and badYears[v,y]==0:
                badYearsforAvg[v,y,cp]=0
        #        if cropYield[icity,y,cp]>0 and cp==0:
        #            goodydcorn+=1
        #        if cropYield[icity,y,cp]>0 and cp==1:
        #            goodydsoy+=1
        #        if cropYield[icity,y,cp]>0 and cp==2:
        #            goodydrice+=1
                
           
    #if goodCity[0,icity]==False:
    #    continue
    #if goodydcorn<20:
    #    #print 'WARNING: NO CORN YIELD DATA'
    #    doCorn=0
    #if goodydsoy<20:
    #    #print 'WARNING: NO SOY YIELD DATA'
    #    doSoy=0
    #if goodydrice<20:
    #    #print 'WARNING: NO RICE YIELD DATA'
    #    doRice=0
    #
    #runCrops[0]=doCorn
    #runCrops[1]=doSoy
    #runCrops[2]=doRice

    ###############################################
    # Make averages
    ###############################################
    mAvg=np.zeros(shape=(2,116,12))  #monthly average
    year=range(1900,2016)
    maxP=-9999*ones(shape=(116))  # the maximum precip in one day for that year
    
    ## compute yearly averages ##
    for v in range(2):
        for y in range(iBeg,iEnd):
            for m in range(12):
                j=0             # j is number of good days in month
                for d in range(31):
                    if dailyData[v,y,m,d]>-100:
                        mAvg[v,y,m]+=dailyData[v,y,m,d]
                        j+=1   
                        if v==2:
                            maxP[y]=max(maxP[y],dailyData[v,y,m,d])
                if v==0 or v==1:
                    mAvg[v,y,m]=mAvg[v,y,m]/j
                if j==0:   # make sure the month has atleast one good day
                    mAvg[v,y,m]=-9999
    
            
    yAvg=np.zeros(shape=(2,116))  #yearly average
    sAvg=np.zeros(shape=(2,116,4))  #seasonal average
        
    ## compute seasonal averages ##
    for v in range(2):
        for y in range(iBeg,iEnd):
            if badYears[v,y]==0:
                if v==0 or v==1:
                    yAvg[v,y]+=sum(mAvg[v,y,:])/12
                else:
                    yAvg[v,y]+=sum(mAvg[v,y,:]) 
            
            if v==0 or v==1:
                sAvg[v,y,0]+=sum(mAvg[v,y,2:5])/3 #spring MAM
                sAvg[v,y,1]+=sum(mAvg[v,y,5:8])/3 #summer JJA
                sAvg[v,y,2]+=sum(mAvg[v,y,8:11])/3 #fall SON
                sAvg[v,y,3]+=sum(mAvg[v,y,0:2]+mAvg[v,y,11])/3 #winter DJF
            if sAvg[0,y,1]>0:
                SummerAvg[icity,y]=sAvg[0,y,1]
      
    ###############################################
    # Find percentiles
    ###############################################
    unsortedD=np.zeros(shape=(4,10970))
    sortedD=np.zeros(shape=(4,10970))
    
    
    for v in range(2):
        i=0
        for y in range(61,90):
            for m in range(12):
                for d in range(31):     
                    if dailyData[v,y,m,d]>-100 and dailyData[v,y,m,d]!=0.00:
                        unsortedD[v,i]=dailyData[v,y,m,d]
                        i+=1

        
        sortedD[v,0:i]=sort(unsortedD[v,0:i])
        i90=int(i*9/10.0)
        i10=int(i/10.)
        
    T90=array([sortedD[0,i90],sortedD[1,i90]]) # 90th percentile
    T10=array([sortedD[0,i10],sortedD[1,i10]]) # 10th percentile
    
    T90all[icity]=T90
    T10all[icity]=T10
    
    ###############################################
    # Find Extremes
    ###############################################
    year=range(1900,2016)
    nyears=max(year)-min(year)+1
    unsortedDall=np.zeros(shape=(4,366*315))
    # initialize variables for the extremes
    T90count=np.zeros(shape=(2,nyears))
    #seasonT90count=zeros(shape=(2,nyears,nCrop))
    T10count=np.zeros(shape=(2,nyears))
    maxT=-9999*ones(shape=(2,nyears))
    minT=9999*ones(shape=(2,nyears))
    frostDays=np.zeros(shape=(nyears))
    #seasonfrostDays=zeros(shape=(nyears,nCrop))
    tropicNights=np.zeros(shape=(nyears))
    #seasontropicNights=zeros(shape=(nyears,nCrop))
    precip95count=np.zeros(shape=(nyears))
    heatwavecount=np.zeros(shape=(2,nyears))
    #seasonheatwavecount=zeros(shape=(2,nyears,nCrop))
    coldwavecount=np.zeros(shape=(2,nyears))
    yr=arange(int(min(year)),int(max(year)+1))
    
    ## compute the extremes ##
    for v in range(2):
        i=3
        daysSinceHeatWave=0
        daysSinceColdWave=0
        for y in range(iBeg,iEnd):
            for m in range(12):
                for d in range(31):
                    if dailyData[v,y,m,d]>-100:
                        unsortedDall[v,i]=dailyData[v,y,m,d]
                        i+=1
                        daysSinceHeatWave+=1
                        daysSinceColdWave+=1    

                        if min(unsortedDall[v,i-3:i])>=T90[v] and daysSinceHeatWave>2:
                            daysSinceHeatWave>2
                            heatwavecount[v,y]+=1
                            daysSinceHeatWave=0
                            #for cp in range(nCrop):
                            #    if runCrops[cp]==0:
                            #        continue
                            #    if m>startGrowingMon[cIndexState[icity],cp] and m<endGrowingMon[cIndexState[icity],cp]:
                            #        seasonheatwavecount[v,y,cp]+=1
                            #    
                            #    elif m==startGrowingMon[cIndexState[icity],cp] and d>=startGrowingDay[cIndexState[icity],cp]:
                            #        seasonheatwavecount[v,y,cp]+=1
                            #        
                            #    elif m==endGrowingMon[cIndexState[icity],cp] and d<=endGrowingDay[cIndexState[icity],cp]:
                            #        seasonheatwavecount[v,y,cp]+=1
        
            if heatwavecount[0,y]>=0:
                HeatWaves[icity,y]=heatwavecount[0,y]

    ###############################################
    # Find GDD and KDD
    ###############################################  
    Tavg=np.zeros(shape=(116,12,31))
    GDD=np.zeros(shape=(116,nCrop))
    #KDD=np.zeros(shape=(116,nCrop))
    KDD=np.zeros(shape=(116))
        
    for y in range(iBeg,iEnd):
        for m in range(12):
            for d in range(31):
                if dailyData[0,y,m,d]==-9999 or dailyData[1,y,m,d]==-9999:
                    continue
                dailytmin=dailyData[1,y,m,d]
                if dailytmin<50:
                    dailytmin=50
                Tavg[y,m,d]=(dailyData[0,y,m,d]+dailytmin)/2
                if Tavg[y,m,d]<68:
                    continue
                
                #for cp in range(nCrop):
                    #if runCrops[cp]==0:
                    #    continue
                    #if m>startGrowingMon[cIndexState[icity],cp] and m<endGrowingMon[cIndexState[icity],cp]:
                KDD[y]+=Tavg[y,m,d]-68
                    
                    #if m==startGrowingMon[cIndexState[icity],cp] and d>=startGrowingDay[cIndexState[icity],cp]:
                    #    KDD[y,cp]+=Tavg[y,m,d]-68  
                    #    
                    #if m==endGrowingMon[cIndexState[icity],cp] and d<=endGrowingDay[cIndexState[icity],cp]:
                    #    KDD[y,cp]+=Tavg[y,m,d]-68                                    
                        
        if KDD[y]>=0:
            KDDays[icity,y]=KDD[y]                   

    ###############################################
    # Make Corr Plots
    ###############################################
    time=True           # iplot=0
    Normalizedtime=True # iplot=1  
    UseKDD=True         # iplot=2
    Summer=True         # iplot=3
    UseHeatWaves=True   # iplot=4
    
    DN=('Days','Nights')
    V=('Max Temp.','Min Temp.','Precip.','Snowfall')
    T=('Tmax','Tmin','precip','snow')
    HL=('Highs','Lows')
    capsDN=('DAYS','NIGHTS')
    capsDNsingle=('DAY','NIGHT')
    #ncrop=('Corn','Soybean')
    ncrop=('Cereals','')
    ncropsave=('cereals','')
    iplot=0
    x=yr[iBeg:iEnd]
    #if not os.path.exists(r'figures/mostcity/'+str(cIndexState[icity])+'/'+city):
    #    os.makedirs(r'figures/mostcity/'+str(cIndexState[icity])+'/'+city)
    if not os.path.exists(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)):
        os.makedirs(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat))
    
    ##                        ##
    ## Make Temperature Plots ##
    ##                        ##
    
    if time: # iplot=0
        cp=0
        #for cp in range(nCrop):
        #    if runCrops[cp]==0:
        #        continue
        x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYear[0,iBeg:iEnd]))
        #ydata=cropYield[icity,iBeg:iEnd,cp]
        ydata=cropYield[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYear[0,iBeg:iEnd]))
        if size(ydata)==0: # don't plot stuff with no data
            continue
            
        if size(x)==0: # don't plot stuff with no data
            continue
            
        ydataAvg=np.mean(ydata)
        slope[iplot,icity,cp],bIntercept[iplot,icity,cp]=polyfit(x,ydata,1)
        yfit=slope[iplot,icity,cp]*x+bIntercept[iplot,icity,cp]
        
        if makePlots:
            #figure(1,figsize=(9,4))
            plt.plot(x,ydata,'--*b',x,yfit,'g')
            plt.ylabel('Yield, Bushels/Acre')
            plt.xlabel('year')
            plt.title(ncrop[cp]+' Yield: '+city+', slope='+str(round(slope[iplot,icity,cp],2))+' Bu/Acre/Year')
            plt.grid(True)
            #savefig('figures/mostcity/'+str(cIndexState[icity])+'/'+city+'/'+ncrop[cp]+'_yield_over_time',dpi=700 )
            plt.savefig(wd+'figures/Ethiopia/'+ncropsave[cp]+'_yield_over_time',dpi=700)
            plt.clf()
    #0
    iplot+=1 #1
    
    if Normalizedtime: # iplot=1
        
        #for cp in range(nCrop):
        #    if runCrops[cp]==0:
        #        continue
            cp=0
            x=yr[iBeg:iEnd]
            ydata=cropYield[iBeg:iEnd]
            if size(ydata[badYearYieldBoolean[0,iBeg:iEnd,cp]])==0: # don't plot stuff with no data
                continue
                
            if size(x)==0: # don't plot stuff with no data
                continue
                
            ydataAvg=np.mean(ydata[badYearYieldBoolean[0,iBeg:iEnd,cp]])
            slope[iplot,icity,cp],b=polyfit(x[badYearYieldBoolean[0,iBeg:iEnd,cp]],ydata[badYearYieldBoolean[0,iBeg:iEnd,cp]],1)
            yfit=slope[iplot,icity,cp]*x+b
            
            num=len(yfit)-1
            
            cropYield[iBeg:iEnd]=ydata-(slope[iplot,icity,cp]*x+b)
            cropYield[iBeg:iEnd]=cropYield[iBeg:iEnd]+yfit[num]
            
            dataAt2015[icity,cp]=yfit[num]
            
            ydata=np.ma.compressed(np.ma.masked_array(cropYield[iBeg:iEnd],badYearYield[0,iBeg:iEnd,cp]))
            x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYearYield[0,iBeg:iEnd,cp]))
            ydataAvg=np.mean(ydata)
            slope[iplot,icity,cp],bIntercept[iplot,icity,cp]=polyfit(x,ydata,1)
            yfit=slope[iplot,icity,cp]*x+bIntercept[iplot,icity,cp]
            Corr[iplot,icity,cp]=corr(x,ydata) 
            
            if makePlots:
                #figure(1,figsize=(9,4))
                plt.plot(x,ydata,'*b',x,yfit,'g')
                plt.ylabel('Yield')
                plt.xlabel('year')
                plt.title(city+' '+ncrop[cp]+' Normalized Yield over time  m='+str(round(slope[iplot,icity,cp],3)*100)+' Corr='+str(round(Corr[iplot,icity,cp],2)))
                plt.grid(True)
                #savefig('figures/mostcity/'+str(cIndexState[icity])+'/'+city+'/'+ncrop[cp]+'_normalized_yield_over_time',dpi=700)
                plt.savefig(wd+'figures/Ethiopia/'+ncropsave[cp]+'_normalized_yield_over_time',dpi=700)
                plt.clf()
    #1
    iplot+=1 #2
    
    if UseKDD: # iplot=2
        #for cp in range(nCrop):
        #    if runCrops[cp]==0:
        #        continue
        cp=0
        x=np.ma.compressed(np.ma.masked_array(KDD[iBeg:iEnd],badYearYield[0,iBeg:iEnd,cp]))
        ydata=cropYield[iBeg:iEnd]
        ydata=ydata[badYearYieldBoolean[0,iBeg:iEnd,cp]]
        if size(ydata)==0: # don't plot stuff with no data
            continue
            
        if size(x)==0: # don't plot stuff with no data
            continue
               
        ydataAvg=np.mean(ydata)
        slope[iplot,icity,cp],bIntercept[iplot,icity,cp]=polyfit(x,ydata,1)
        yfit=slope[iplot,icity,cp]*x+bIntercept[iplot,icity,cp]
        Corr[iplot,icity,cp]=corr(x,ydata)
        R2[iplot,icity,cp]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg) 
        
        if makePlots:
            #figure(1,figsize=(9,4))
            plt.plot(x,ydata,'*b',x,yfit,'g')
            plt.ylabel(ncrop[cp]+' Yield, Bushels/Acre')
            plt.xlabel('Killing Degree Days per Growing Season')
            plt.title(ncrop[cp]+' Yield vs Killing Degree Days, '+city+' IL, Corr='+str(round(Corr[iplot,icity,cp],2)))
            plt.grid(True)
            #savefig('figures/mostcity/'+str(cIndexState[icity])+'/'+city+'/'+ncrop[cp]+'_KDD_yield_corr',dpi=700 )
            plt.savefig(wd+'figures/Ethiopia/'+ncropsave[cp]+'_KDD_yield_corr',dpi=700)
            plt.clf()
    #2
    iplot+=1 #3
    
    if Summer: # iplot=3
        #for cp in range(nCrop):
        #    if runCrops[cp]==0:
        #        continue
        ydata=cropYield[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYearsforAvg[0,iBeg:iEnd,cp])) 
        x=sAvg[0,iBeg:iEnd,1]
        #for d in range(len(x)):
        #    x[d]=(x[d]-32.)*(5./9.)
        x=np.ma.compressed(np.ma.masked_array(x,badYearsforAvg[0,iBeg:iEnd,cp]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
            
        if size(x)==0: # don't plot stuff with no data
            continue
            
        ydataAvg=np.mean(ydata)
        slope[iplot,icity,cp],bIntercept[iplot,icity,cp]=polyfit(x,ydata,1)
        yfit=slope[iplot,icity,cp]*x+bIntercept[iplot,icity,cp]
        Corr[iplot,icity,cp]=corr(x,ydata) 
        R2[iplot,icity,cp]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)

        if makePlots:
            #figure(1,figsize=(9,4))
            plt.plot(x,ydata,'*b',x,yfit,'g')
            plt.ylabel(ncrop[cp]+' Yield, Bushels/Acre')
            plt.xlabel('Avg Summer (JJA) Temp, C')
            plt.title(ncrop[cp]+' Yield vs Summer Avg Temp, '+city+', Corr='+str(round(Corr[iplot,icity,cp],2)))
            plt.grid(True)
            #savefig('figures/mostcity/'+str(cIndexState[icity])+'/'+city+'/'+ncrop[cp]+'_summertemp_yield_corr_'+city+'_C',dpi=700 )
            plt.savefig(wd+'figures/Ethiopia/'+ncropsave[cp]+'_summertemp_yield_corr_'+city,dpi=700)
            plt.clf()
    #3
    iplot+=1 #4
    
              
    if UseHeatWaves: # iplot=4
        #for cp in range(nCrop):
        #    if runCrops[cp]==0:
        #        continue
        ydata=cropYield[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYearsforAvg[0,iBeg:iEnd,cp]))
        x=heatwavecount[0,iBeg:iEnd]
        x=np.ma.compressed(np.ma.masked_array(x,badYearsforAvg[0,iBeg:iEnd,cp]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
            
        if size(x)==0: # don't plot stuff with no data
            continue
            
        ydataAvg=np.mean(ydata)
        slope[iplot,icity,cp],bIntercept[iplot,icity,cp]=polyfit(x,ydata,1)
        yfit=slope[iplot,icity,cp]*x+bIntercept[iplot,icity,cp]
        Corr[iplot,icity,cp]=corr(x,ydata) 
        R2[iplot,icity,cp]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)
        
        if makePlots:
            plt.figure(1,figsize=(9,4))
            plt.plot(x,ydata,'*b',x,yfit,'g')
            plt.ylabel(ncrop[cp]+' Yield, Bushels/Acre')
            plt.xlabel('Heat Waves, Number/Growing Season')
            plt.title(ncrop[cp]+' Yield vs Heat Waves, '+city+', Corr='+str(round(Corr[iplot,icity,cp],2)))
            plt.grid(True)
            #savefig('figures/mostcity/'+str(cIndexState[icity])+'/'+city+'/'+ncrop[cp]+'_heat_waves_yield_corr_'+city+'_little',dpi=700 )
            plt.savefig(wd+'figures/Ethiopia/'+ncropsave[cp]+'_heat_waves_yield_corr_'+city,dpi=700)
            plt.clf()

    #for cp in range(nCrop):
        #if runCrops[cp]==0:
        #        continue
    for iplot in range(5):
        cp=0
        if iplot==0:
            maxCorr[icity,cp]=Corr[iplot,icity,cp]
            maxCorrAbs=abs(Corr[iplot,icity,cp])
            iplotCorr[icity,cp]=iplot
            maxCorrAbs2=abs(Corr[iplot,icity,cp])
            
        if abs(Corr[iplot,icity,cp])>maxCorrAbs:
            maxCorr2[icity,cp]=maxCorr[icity,cp]
            maxCorrAbs2=abs(maxCorr2[icity,cp])
            iplotCorr2[icity,cp]=iplotCorr[icity,cp]
            
            maxCorr[icity,cp]=Corr[iplot,icity,cp]
            maxCorrAbs=abs(Corr[iplot,icity,cp])
            iplotCorr[icity,cp]=iplot
            
        if maxCorrAbs2<abs(Corr[iplot,icity,cp])<maxCorrAbs:
            maxCorr2[icity,cp]=Corr[iplot,icity,cp]
            maxCorrAbs2=abs(Corr[iplot,icity,cp])
            iplotCorr2[icity,cp]=iplot
                
    if makePlots:
        figure(1,figsize=(9,4))
        #                                   ##
        ## Summer Avg Correlation with Yield ##
        ##                                   ##
        ticks=([110,120,130,140,150,160,170,180,190,200,210,220],[35,40,45,50,55,60])
        
        #x=np.arange(1970,2015)
        x=np.arange(iBeg,iEnd)
        ydata=sAvg[0,iBeg:iEnd,1]
        
        #x2=range(1970,2015)
        x2=np.arange(iBeg,iEnd)
        ydata2=cropYield[iBeg:iEnd]
        
        fig, ax1 = plt.subplots()
        
        ax2 = ax1.twinx()
        ax1.plot(x,ydata,'-*g')
        ax2.plot(x2,ydata2,'-*b')
        #ax1.set_yticks([70,75,80,85,90])
        #ax2.set_yticks([100,120,140,160,180,200,220,240,260,280])
        ax2.set_ylabel('Detrended Corn Yield, Bushels/Acre',color='b')
        ax1.set_ylabel('Summer Avg Temperature, F',color='g')
        #ax1.set(color='g')
        #ax2.set(color='b')
        ax1.set_xlabel('year')
        title(ncrop[0]+' Yield and Summer Avg Temp, '+city+', IL, Corr='+str(round(Corr[3,icity,0],2)))
        #plt.minorticks_on()
        ax1.add_line(Line2D([1980,1980],[50,100],linewidth=.5,color='k',linestyle='dashed'))
        ax1.add_line(Line2D([1983,1983],[50,100],linewidth=.5,color='k',linestyle='dashed'))
        ax1.add_line(Line2D([1988,1988],[50,100],linewidth=.5,color='k',linestyle='dashed'))
        ax1.add_line(Line2D([1991,1991],[50,100],linewidth=.5,color='k',linestyle='dashed'))
        ax1.add_line(Line2D([1995,1995],[50,100],linewidth=.5,color='k',linestyle='dashed'))
        ax1.add_line(Line2D([2002,2002],[50,100],linewidth=.5,color='k',linestyle='dashed'))
        ax1.add_line(Line2D([2012,2012],[50,100],linewidth=.5,color='k',linestyle='dashed'))
        #savefig('figures/mostcity/'+str(cIndexState[icity])+'/'+city+'/'+ncrop[0]+'_corr_with_summer_avg_little',dpi=700 )
        plt.clf()
    
    #print 'Corn:'      
    #print round(maxCorr[icity,0],2),round(R2[iplotCorr[icity,0],icity,0],2),iplotCorr[icity,0]
    #print round(maxCorr2[icity,0],2),round(R2[iplotCorr2[icity,0],icity,0],2),iplotCorr2[icity,0],'\n' 
    #print 'Soybeans:'      
    #print round(maxCorr[icity,1],2),round(R2[iplotCorr[icity,0],icity,1],2),iplotCorr[icity,1]
    #print round(maxCorr2[icity,1],2),round(R2[iplotCorr2[icity,0],icity,1],2),iplotCorr2[icity,1],'\n'
    #print 'Rice:'      
    #print round(maxCorr[icity,2],2),round(R2[iplotCorr[icity,0],icity,2],2),iplotCorr[icity,2]
    #print round(maxCorr2[icity,2],2),round(R2[iplotCorr2[icity,0],icity,2],2),iplotCorr2[icity,2],'\n'

####################################################################
# Write slope, corr, lat, lon, station name into a text document
####################################################################
#if not os.path.exists(r'final_data/'):
#    os.makedirs(r'final_data/')
#pickle.dump(dataAt2015,open('pickle_files/dataAt2015.p','wb'))
#pickle.dump(slope,open('pickle_files/slope.p','wb'))
#pickle.dump(bIntercept,open('pickle_files/bIntercept.p','wb'))
#pickle.dump(lat,open('pickle_files/lat.p','wb'))
#pickle.dump(lon,open('pickle_files/lon.p','wb'))
#pickle.dump(stationList,open('pickle_files/station.p','wb'))
#pickle.dump(goodCity,open('pickle_files/goodCity.p','wb'))
#pickle.dump(cityList,open('pickle_files/cityList.p','wb'))
#pickle.dump(Corr,open('pickle_files/Corr.p','wb'))
#pickle.dump(maxCorr,open('pickle_files/maxCorr.p','wb'))
#pickle.dump(iplotCorr,open('pickle_files/iplotCorr.p','wb'))
#pickle.dump(maxCorr2,open('pickle_files/maxCorr2.p','wb'))
#pickle.dump(iplotCorr2,open('pickle_files/iplotCorr2.p','wb'))
#pickle.dump(T90all,open('pickle_files/T90all.p','wb'))
#pickle.dump(T10all,open('pickle_files/T10all.p','wb'))
#'''
#pickle.dump(SummerAvg,open('pickle_files/SummerAvg.p','wb'))
#pickle.dump(HeatWaves,open('pickle_files/HeatWaves.p','wb'))
#pickle.dump(KDDays,open('pickle_files/KDD.p','wb'))
#'''
#pickle.dump(cropYield,open('pickle_files/NormalizedCropYield.p','wb'))

###############################################
# The Main Code
# Reads in data
# gets rid of bad cities
# computes the means and extremes
# plots and records slope, R2 value, lat, lon, station name
###############################################
from pylab import *
import csv
from math import sqrt
from sys import exit
import numpy as np
import pickle
import os

#cd \Users\lilli_000\Documents\Science_Fair_2016_Climate_Change\code

# choose what you want the code to run for #
makePlots=True
precip=False
temp=True
useUniformStartDate=False
dataYears=150
goodCityRatio=.85

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
    xAvg=Avg(x)
    xOut=0.
    for k in range(len(x)):
        xOut=xOut+(x[k]-xAvg)**2
    xOut=xOut/(k+1)
    xOut=sqrt(xOut)
    return xOut
    
def Variance(x):   
    '''function to compute the variance (std dev squared)'''
    xAvg=Avg(x)
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
    
###############################################
# Read in Weather Data
###############################################
if temp:
    cityFile='atlantic_city'
    #cityFile='tmax_cities_over_'+str(dataYears)+'_years_no_usc' # what file to read in #
    
     #decide the number of cities
    if dataYears==150:
        nCities=14
    if dataYears==130:
        nCities=92
    if dataYears==100:
        nCities=2129
    if dataYears==65:
        nCities=4534
    if dataYears==120:
        nCities=911
        
if precip:
    cityFile='precip_cities_over_'+str(dataYears)+'_years_no_usc' # what file to read in #
    
    # decide the number of cities
    if dataYears==150:
        nCities=47
    if dataYears==130:
        nCities=658
    if dataYears==100:
        nCities=4748
    if dataYears==65:
        nCities=9276
    if dataYears==120:
        nCities=2235
    
f=open('written_files/'+cityFile+'.txt','r')

# initialize variables
icity=-1 # a counter of the number of cities
slope=-9999.*ones(shape=(22,nCities))
R2=-9999.*ones(shape=(22,nCities))
lat=-9999.*ones(shape=(nCities))
lon=-9999.*ones(shape=(nCities))
stationList=[]
cityList=[]
goodCity=ones(shape=(4,nCities),dtype=bool)

for cityline in f:  # read in the file written in the other code
    print icity, ' of ', nCities
    icity+=1
    # initialize more variables
    tmp=cityline[0:75]
    station=tmp[0:11]
    stationList.append(station)
    lat[icity]=tmp[12:20]
    lon[icity]=tmp[21:30]
    iBeg=int(tmp[36:40])-1700
    if useUniformStartDate:
        iBeg=(2015-dataYears)-1700
    iEnd=int(tmp[41:45])-1700
    city=tmp[46:75].strip()
    titleCity=city
    cityList.append(titleCity)
    
    city=city.replace(' ','_')
    if tmp[0:3]=='USC': # USC stations have bad data so don't use them
        goodCity[:,icity]=False 
        print 'station: ',station,' is a USC'
        continue
   
    # initialize more variables
    dailyData=-9999.*ones(shape=(4,316,12,31))
    badData=zeros(shape=(4,316))  
    badYears=ones(shape=(4,316))
        
    f = open('data/noaa_daily_data/ghcnd_all/'+station+'.dly', 'r')
    
    for line in f: #read in the daily data for each station
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
        elif var=='PRCP':
            v=2
        elif var=='SNOW':
            v=3
        else:
            v=-1
        
        if var!=-1:   
            m=month-1
            y=year-1700
            d=21
            for d in range(31):
                tmp2=tmp[21+8*d:21+8*d+5]
                if tmp2=='   0T' or tmp2=='   0P':
                    tmp2='    0'
                if tmp2!='-9999':
                    if v==0 or v==1:
                        tmp3=(9./5.*float(tmp2)/10.+32.)
                    else:
                        tmp3=(float(tmp2)/10/10/2.54)
                else:
                    badData[v,y]+=1
                    tmp3=-9999
                dailyData[v,y,m,d]=tmp3 # put the data into one 4D array
    
    ###############################################
    # Make averages
    ###############################################
    mAvg=zeros(shape=(4,316,12))  #monthly average
    year=range(1700,2016)
    maxP=-9999*ones(shape=(316))  # the maximum precip in one day for that year
    
    ## compute yearly averages ##
    for v in range(3): 
        for y in range(iBeg,iEnd):
            goodMonths=0
            for m in range(12):
                j=0             # j is number of good days in month
                for d in range(31):
                    if dailyData[v,y,m,d]>-100:
                        mAvg[v,y,m]+=dailyData[v,y,m,d]
                        j+=1   
                        if v==2:
                            maxP[y]=max(maxP[y],dailyData[v,y,m,d])
                if j>0:
                    goodMonths+=1
                    if v==0 or v==1:
                        mAvg[v,y,m]=mAvg[v,y,m]/j
                if j==0:   # make sure the month has atleast one good day
                    mAvg[v,y,m]=-9999
                if goodMonths==12:    #if all 12 months are good, keep the year
                    badYears[v,y]=0
    
        if sum(badYears[v,iBeg:iEnd])/dataYears > goodCityRatio:
            goodCity[v,icity]=False
            
    yAvg=zeros(shape=(4,316))  #yearly average
    sAvg=zeros(shape=(4,316,4))  #seasonal average
    
    ## compute seasonal averages ##
    for v in range(3):
        for y in range(316):
            if v==0 or v==1:
                yAvg[v,y]+=sum(mAvg[v,y,:])/12
            else:
                yAvg[v,y]+=sum(mAvg[v,y,:]) 
            
            if v==0 or v==1:
                sAvg[v,y,0]+=sum(mAvg[v,y,2:5])/3 #spring MAM
                sAvg[v,y,1]+=sum(mAvg[v,y,5:8])/3 #summer JJA
                sAvg[v,y,2]+=sum(mAvg[v,y,8:11])/3 #fall SON
                sAvg[v,y,3]+=sum(mAvg[v,y,0:2]+mAvg[v,y,11])/3 #winter DJF
            else:
                sAvg[v,y,0]+=sum(mAvg[v,y,2:5]) #spring MAM
                sAvg[v,y,1]+=sum(mAvg[v,y,5:8]) #summer JJA
                sAvg[v,y,2]+=sum(mAvg[v,y,8:11]) #fall SON
                sAvg[v,y,3]+=sum(mAvg[v,y,0:2]+mAvg[v,y,11]) #winter DJF
                
          
    ###############################################
    # Find percentiles
    ###############################################
    unsortedD=zeros(shape=(4,10970))
    sortedD=zeros(shape=(4,10970))
    
    
    for v in range(3):
        i=0
        p=0
        for y in range(261,290):
            for m in range(12):
                for d in range(31):     
                    if dailyData[v,y,m,d]>-100:
                        unsortedD[v,i]=dailyData[v,y,m,d]
                        i+=1
                        if v==2:
                            p+=1

        
        sortedD[v,0:i]=sort(unsortedD[v,0:i])
        i90=int(i*9/10.0)
        i10=int(i/10.)
        p95=int(p*9.5/10)
        
    T90=array([sortedD[0,i90],sortedD[1,i90]]) # 90th percentile
    T10=array([sortedD[0,i10],sortedD[1,i10]]) # 10th percentile
    precip95=array([sortedD[2,p95]])  # 95th percentile for precip
    
    ###############################################
    # Find Extremes
    ###############################################
    year=range(1700,2016)
    nyears=max(year)-min(year)+1
    unsortedDall=zeros(shape=(4,366*315))
    # initialize variables for the extremes
    maxDaysSinceRain=zeros(shape=(nyears))
    max5DayPrecip=zeros(shape=(nyears))
    T90count=zeros(shape=(2,nyears))
    T10count=zeros(shape=(2,nyears))
    maxT=-9999*ones(shape=(2,nyears))
    minT=9999*ones(shape=(2,nyears))
    maxP=zeros(shape=(nyears))
    frostDays=zeros(shape=(nyears))
    tropicNights=zeros(shape=(nyears))
    precip95count=zeros(shape=(nyears))
    heatwavecount=zeros(shape=(2,nyears))
    coldwavecount=zeros(shape=(2,nyears))
    yr=arange(int(min(year)),int(max(year)+1))
    daysSinceRain=0
    
    ## compute the extremes ##
    for v in range(3): 
        i=3
        daysSinceHeatWave=0
        daysSinceColdWave=0
        for y in range(iBeg,iEnd):
            for m in range(12):
                for d in range(31):
                    if dailyData[v,y,m,d]>-100 and v<2:
                        unsortedDall[v,i]=dailyData[v,y,m,d]
                        i+=1
                        daysSinceHeatWave+=1
                        daysSinceColdWave+=1    
                        if dailyData[v,y,m,d]>=T90[v]:
                            T90count[v,y]+=1
                        if dailyData[v,y,m,d]<=T10[v]:
                            T10count[v,y]+=1
                        if min(unsortedDall[v,i-3:i])>=T90[v] and daysSinceHeatWave>2:
                            daysSinceHeatWave>2
                            heatwavecount[v,y]+=1
                            daysSinceHeatWave=0
                        if max(unsortedDall[v,i-3:i])<=T10[v]and daysSinceColdWave>2:
                            coldwavecount[v,y]+=1
                            daysSinceColdWave=0 
                        if dailyData[v,y,m,d]>-100:
                            maxT[v,y]=max(maxT[v,y],dailyData[v,y,m,d])
                            minT[v,y]=min(minT[v,y],dailyData[v,y,m,d])
                        if v==1 and -100<dailyData[v,y,m,d]<32:
                            frostDays[y]+=1
                        if v==1 and dailyData[v,y,m,d]>68:
                            tropicNights[y]+=1
                    if dailyData[v,y,m,d]>-100 and v==2: 
                        if dailyData[v,y,m,d]>=precip95:
                           precip95count[y]+=dailyData[v,y,m,d]
                        if dailyData[v,y,m,d]<0.04:
                            daysSinceRain+=1
                        else:
                            maxDaysSinceRain[y]=max(maxDaysSinceRain[y],daysSinceRain)
                            daysSinceRain=0
                        max5DayPrecip[y]=max(max5DayPrecip[y],sum(dailyData[v,y,m,d-5:d]))
                        maxP[y]=np.amax(dailyData[v,y,:,:])
    
    ###############################################
    # Make Plots
    ###############################################
    DN=('Days','Nights')
    V=('Max Temp.','Min Temp.','Precip.','Snowfall')
    T=('Tmax','Tmin','precip','snow')
    HL=('Highs','Lows')
    capsDN=('DAYS','NIGHTS')
    capsDNsingle=('DAY','NIGHT')
    iplot=0
    x=yr[iBeg:iEnd]
    if makePlots:  # only make plots if the makePlots variable = true    
        if not os.path.exists(r'figures/'+cityFile+'/'+city):
            os.makedirs(r'figures/'+cityFile+'/'+city)

    ##                        ##
    ## Make Temperature Plots ##
    ##                        ##
    if temp:
         ## Yearly temperature Averages ##    
        for v in range(2):
            x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[v,iBeg:iEnd]))
            ydata=yAvg[v,iBeg:iEnd]
            ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[v,iBeg:iEnd]))
            if size(ydata)==0: # don't plot stuff with no data
                continue
                
            if size(x)==0: # don't plot stuff with no data
                continue
                
            ydataAvg=Avg(ydata)
            slope[iplot,icity],b=polyfit(x,ydata,1)
            yfit=slope[iplot,icity]*x+b
            R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg) 
            
            if makePlots:  # only make plots if the makePlots variable = true 
                #figure(1,figsize=(30,30))
                plot(x,ydata,'*b',x,yfit,'g')
                xlabel('Year')
                ylabel('Yearly Average of Daily '+V[v]+', F')
                title(titleCity+' YEARLY AVERAGE of '+HL[v]+'  m='+str(round(slope[iplot,icity],3)*100)+
                    ' R2='+str(round(R2[iplot,icity],2)))
                grid(True)
                savefig('figures/'+cityFile+'/'+city+'/test/'+city+'_'+T[v]+'_yearly_avg_squshed.png',
                    dpi=700 )
                show()
                exit()
                clf()
            iplot+=1
            # Tmax iplot=0
            # Tmin iplot=1
            
        ## Warm Days and Warm Nights ##
        for v in range(2):
            x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[v,iBeg:iEnd]))
            ydata=T90count[v,iBeg:iEnd]
            ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[v,iBeg:iEnd]))
            
            if size(ydata)==0: # don't plot stuff with no data
                continue
            if size(x)==0: # don't plot stuff with no data
                continue
            ydataAvg=Avg(ydata)
            slope[iplot,icity],b=polyfit(x,ydata,1)
            yfit=slope[iplot,icity]*x+b
            R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg) 
            
            if makePlots:  # only make plots if the makePlots variable = true            
                plot(x,ydata,'*b',x,yfit,'g')
                xlabel('Year')
                ylabel('# of '+DN[v]+' Exceeding the 90th %: '+str(T90[v])+' F')
                title(titleCity+' WARM '+capsDN[v]+'  m='+str(round(slope[iplot,icity],3)*100)+
                    ' R2='+str(round(R2[iplot,icity],2)))
                grid(True)
                savefig('figures/'+cityFile+'/'+city+'/'+city+'_warm_'+DN[v]+'_squshed.png')
                show()
                clf()
            iplot+=1
            # Warm days iplot=2
            # Warm nights iplot=3
        
        ## Cold Days and Cold Nights ##   
        for v in range(2):
            x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[v,iBeg:iEnd]))
            ydata=T10count[v,iBeg:iEnd]
            ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[v,iBeg:iEnd]))
            
            if size(ydata)==0: # don't plot stuff with no data
                continue
            if size(x)==0: # don't plot stuff with no data
                continue
            ydataAvg=Avg(ydata)
            slope[iplot,icity],b=polyfit(x,ydata,1)
            yfit=slope[iplot,icity]*x+b
            R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg) 
    
            if makePlots:  # only make plots if the makePlots variable = true
                plot(x,ydata,'*b',x,yfit,'g')
                xlabel('Year')
                ylabel('# of '+DN[v]+' Below the 10th %: '+str(T10[v])+' F')
                title(titleCity+' COLD '+capsDN[v]+'  m='+str(round(slope[iplot,icity],3)*100)+
                    ' R2='+str(round(R2[iplot,icity],2)))
                grid(True)
                savefig('figures/'+cityFile+'/'+city+'/'+city+'_cold_'+DN[v]+'_squshed.png')
                show()
                clf()
            iplot+=1
            # Cold days iplot=4
            # Cold nights iplot=5
    
        
        ## Heat Waves ## 
        for v in range(2):  
            x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[v,iBeg:iEnd]))
            ydata=heatwavecount[v,iBeg:iEnd]
            ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[v,iBeg:iEnd]))
            
            if size(ydata)==0: # don't plot stuff with no data
                continue
            if size(x)==0: # don't plot stuff with no data
                continue
            ydataAvg=Avg(ydata)
            slope[iplot,icity],b=polyfit(x,ydata,1)
            yfit=slope[iplot,icity]*x+b
            R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg) 
            
            if makePlots:  # only make plots if the makePlots variable = true
                plot(x,ydata,'*b',x,yfit,'g')
                xlabel('Year')
                ylabel('# Heat Waves(3 cons. days >90th %: '+str(T90[v])+')')
                title(titleCity+' HEAT WAVES of '+HL[v]+'  m='+str(round(slope[iplot,icity],3)*100)+
                    ' R2='+str(round(R2[iplot,icity],2)))
                grid(True)
                savefig('figures/'+cityFile+'/'+city+'/'+city+'_'+T[v]+'_heat_waves_squshed.png')
                show()
                clf()
            iplot+=1
            # Heat waves of highs=6
            # Heat waves of lows=7
                
        ## Cold Spells ##
        for v in range(2):
            x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[v,iBeg:iEnd]))
            ydata=coldwavecount[v,iBeg:iEnd]
            ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[v,iBeg:iEnd]))
            
            if size(ydata)==0: # don't plot stuff with no data
                continue
            if size(x)==0: # don't plot stuff with no data
                continue
            ydataAvg=Avg(ydata)
            slope[iplot,icity],b=polyfit(x,ydata,1)
            yfit=slope[iplot,icity]*x+b
            R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)
            
            if makePlots:  # only make plots if the makePlots variable = true
                plot(x,ydata,'*b',x,yfit,'g')
                xlabel('Year')
                ylabel('# Cold Spells(3 cons. days <10th %: '+str(T10[v])+')')
                title(titleCity+' COLD SPELLS of '+HL[v]+'  m='+str(round(slope[iplot,icity],3)*100)+
                    ' R2='+str(round(R2[iplot,icity],2)))
                grid(True)
                savefig('figures/'+cityFile+'/'+city+'/'+city+'_'+T[v]+'_cold_spells_squshed.png')
                show()
                clf()
            iplot+=1
            # Cold Spells of highs=8
            # Cold Spells of lows=9
            
            ## Warmest Daily Temperature ##    
        for v in range(2):
            x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[v,iBeg:iEnd]))
            ydata=maxT[v,iBeg:iEnd]
            ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[v,iBeg:iEnd]))
            
            if size(ydata)==0: # don't plot stuff with no data
                continue
            if size(x)==0: # don't plot stuff with no data
                continue
            ydataAvg=Avg(ydata)
            slope[iplot,icity],b=polyfit(x,ydata,1)
            yfit=slope[iplot,icity]*x+b
            R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)
            
            if makePlots:  # only make plots if the makePlots variable = true
                plot(x,ydata,'*b',x,yfit,'g')
                xlabel('Year')
                ylabel('Warmest Temperature of the Year, F')
                title(titleCity+' WARMEST '+capsDNsingle[v]+'  m='+str(round(slope[iplot,icity],3)*100)+
                    ' R2='+str(round(R2[iplot,icity],2)))
                grid(True)
                savefig('figures/'+cityFile+'/'+city+'/'+city+'_warmest_'+T[v]+'_squshed.png')
                show()
                clf()
            iplot+=1
            # Warmest Daily High=10
            # Warmest Daily Low=11
            
        ## Coldest Daily Temperature ##    
        for v in range(2):
            x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[v,iBeg:iEnd]))
            ydata=minT[v,iBeg:iEnd]
            ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[v,iBeg:iEnd]))
            
            if size(ydata)==0: # don't plot stuff with no data
                continue
            if size(x)==0: # don't plot stuff with no data
                continue
            ydataAvg=Avg(ydata)
            slope[iplot,icity],b=polyfit(x,ydata,1)
            yfit=slope[iplot,icity]*x+b
            R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)
            
            if makePlots:  # only make plots if the makePlots variable = true
                plot(x,ydata,'*b',x,yfit,'g')
                xlabel('Year')
                ylabel('Coldest Temperature of the Year, F')
                title(titleCity+' COLDEST '+capsDNsingle[v]+'  m='+str(round(slope[iplot,icity],3)*100)+
                    ' R2='+str(round(R2[iplot,icity],2)))
                grid(True)
                savefig('figures/'+cityFile+'/'+city+'/'+city+'_coldest_'+T[v]+'_squshed.png')
                show()
                clf()
            iplot+=1
            # Coldest Daily High=12
            # Coldest Daily Low=13
            
            ## Frost Nights ##  
        x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[1,iBeg:iEnd])) 
        ydata=frostDays[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[1,iBeg:iEnd]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
        if size(x)==0: # don't plot stuff with no data
            continue
        ydataAvg=Avg(ydata)
        slope[iplot,icity],b=polyfit(x,ydata,1)
        yfit=slope[iplot,icity]*x+b
        R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg) 
        
        if makePlots:  # only make plots if the makePlots variable = true
            plot(x,ydata,'*b',x,yfit,'g')
            xlabel('Year')
            ylabel('Number of Nights Below 32 F')
            title(titleCity+' FROST NIGHTS'+'  m='+str(round(slope[iplot,icity],3)*100)+
                ' R2='+str(round(R2[iplot,icity],2)))
            grid(True)
            savefig('figures/'+cityFile+'/'+city+'/'+city+'_frost_nights_squshed.png')
            show()
            clf()
        iplot+=1
        # Frost Nights=14
        
        ## Tropical Nights ##
        x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[1,iBeg:iEnd]))
        ydata=tropicNights[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[1,iBeg:iEnd]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
        if size(x)==0: # don't plot stuff with no data
            continuey
        dataAvg=Avg(ydata)
        slope[iplot,icity],b=polyfit(x,ydata,1)
        yfit=slope[iplot,icity]*x+b
        R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)  
        
        if makePlots:  # only make plots if the makePlots variable = true
            plot(x,ydata,'*b',x,yfit,'g')
            xlabel('Year')
            ylabel('Number of Nights Above 68 F')
            title(titleCity+' TROPICAL NIGHTS'+'  m='+str(round(slope[iplot,icity],3)*100)+
                ' R2='+str(round(R2[iplot,icity],2)))
            grid(True)
            savefig('figures/'+cityFile+'/'+city+'/'+city+'_tropic_nights_squshed.png')
            show()
            clf()
        iplot+=1
        # Tropical Nights=15
            
            
    ##                   ##
    ## Make Precip Plots ##
    ##                   ##
    if precip:    
        ## Yearly precipitation Sums ##    
        x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[2,iBeg:iEnd]))
        ydata=yAvg[v,iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[2,iBeg:iEnd]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
        if size(x)==0: # don't plot stuff with no data
            continue
        ydataAvg=Avg(ydata)
        ydataAvg=Avg(ydata)
        slope[iplot,icity],b=polyfit(x,ydata,1)
        yfit=slope[iplot,icity]*x+b
        R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)
        
        if makePlots:  # only make plots if the makePlots variable = true
            figure(1,figsize=(9,3.8))
            plot(x,ydata,'*b',x,yfit,'g')
            xlabel('Year')
            ylabel('Total Yearly Precip., in')
            title(titleCity+' TOTAL YEARLY PRECIPITATION'+'  m='+str(round(slope[iplot,icity],3)*100)+
                ' R2='+str(round(R2[iplot,icity],2)))
            grid(True)
            savefig('figures/'+cityFile+'/'+city+'/'+city+'_'+T[v]+'_yearly_avg_squshed.png')
            show()
            clf()
        iplot+=1
        
        ## Wettest Days ## 
        x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[2,iBeg:iEnd]))
        ydata=precip95count[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[2,iBeg:iEnd]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
        if size(x)==0: # don't plot stuff with no data
            continue
        ydataAvg=Avg(ydata)
        slope[iplot,icity],b=polyfit(x,ydata,1)
        yfit=slope[iplot,icity]*x+b
        R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)  
        
        if makePlots:  # only make plots if the makePlots variable = true
            plot(x,ydata,'*b',x,yfit,'g')
            xlabel('Year')
            ylabel('Total Yearly Precip, days >95th %: '+str(round(precip95,2))+' ,in')
            title(titleCity+' WETTEST DAYS'+'  m='+str(round(slope[iplot,icity],3)*100)+
                ' R2='+str(round(R2[iplot,icity],2)))
            grid(True)
            savefig('figures/'+cityFile+'/'+city+'/'+city+'_precip_extremes_squshed.png')
            show()
            clf()
        iplot+=1
    
        ## Dry Spells ##
        x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[2,iBeg:iEnd]))
        ydata=maxDaysSinceRain[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[2,iBeg:iEnd]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
        if size(x)==0: # don't plot stuff with no data
            continue
        ydataAvg=Avg(ydata)
        slope[iplot,icity],b=polyfit(x,ydata,1)
        yfit=slope[iplot,icity]*x+b
        R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)  
        
        if makePlots:  # only make plots if the makePlots variable = true
            figure(1)
            plot(x,ydata,'*b',x,yfit,'g')
            xlabel('Year')
            ylabel('Max # of Cons. Days W/o Significant Rain')
            title(titleCity+' DRY SPELLS'+'  m='+str(round(slope[iplot,icity],3)*100)+
                ' R2='+str(round(R2[iplot,icity],2)))
            grid(True)
            savefig('figures/'+cityFile+'/'+city+'/'+city+'_droughts_squshed.png')
            show()
            clf()
        iplot+=1
    
        ## Wet Spells ##
        x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[2,iBeg:iEnd]))
        ydata=max5DayPrecip[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[2,iBeg:iEnd]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
        if size(x)==0: # don't plot stuff with no data
            continue
        ydataAvg=Avg(ydata)
        slope[iplot,icity],b=polyfit(x,ydata,1)
        yfit=slope[iplot,icity]*x+b
        R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg)
        
        if makePlots:  # only make plots if the makePlots variable = true
            figure(1)
            plot(x,ydata,'*b',x,yfit,'g')
            xlabel('Year')
            ylabel('Max Precip in a 5 Day Period, in')
            title(titleCity+' WET SPELLS'+'  m='+str(round(slope[iplot,icity],3)*100)+
                ' R2='+str(round(R2[iplot,icity],2)))
            grid(True)
            savefig('figures/'+cityFile+'/'+city+'/'+city+'_heavy_rain_squshed.png')
            show()
            clf()
        iplot+=1
    
        ## Wettest Day ##
        x=np.ma.compressed(np.ma.masked_array(yr[iBeg:iEnd],badYears[2,iBeg:iEnd]))
        ydata=maxP[iBeg:iEnd]
        ydata=np.ma.compressed(np.ma.masked_array(ydata,badYears[2,iBeg:iEnd]))
        
        if size(ydata)==0: # don't plot stuff with no data
            continue
        if size(x)==0: # don't plot stuff with no data
            continue
        ydataAvg=Avg(ydata)
        slope[iplot,icity],b=polyfit(x,ydata,1)
        yfit=slope[iplot,icity]*x+b
        R2[iplot,icity]=1.-SumOfSquares(yfit-ydata)/SumOfSquares(ydata-ydataAvg) 
        
        if makePlots:  # only make plots if the makePlots variable = true
            figure(1)
            plot(x,ydata,'*b',x,yfit,'g')
            xlabel('Year')
            ylabel('Wettest Day')
            title(titleCity+' WETTEST DAY, in'+'  m='+str(round(slope[iplot,icity],3)*100)+
                ' R2='+str(round(R2[iplot,icity],2)))
            grid(True)
            savefig('figures/'+cityFile+'/'+city+'/'+city+'_wettest_day_squshed.png')
            show()
            clf()
        iplot+=1
    if icity==1:
        exit()

####################################################################
# Write slope, R2 value, lat, lon, station name into a text document
####################################################################
if not os.path.exists(r'final_data/'+cityFile):
    os.makedirs(r'final_data/'+cityFile)
pickle.dump(100*slope,open('final_data/'+cityFile+'/slope.p','wb'))
pickle.dump(R2,open('final_data/'+cityFile+'/R2.p','wb'))
pickle.dump(lat,open('final_data/'+cityFile+'/lat.p','wb'))
pickle.dump(lon,open('final_data/'+cityFile+'/lon.p','wb'))
pickle.dump(stationList,open('final_data/'+cityFile+'/station.p','wb'))
pickle.dump(goodCity,open('final_data/'+cityFile+'/goodCity.p','wb'))
pickle.dump(cityList,open('final_data/'+cityFile+'/cityList.p','wb'))
################################################
# The Forth Code
# Makes CSV files
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
    
def corr(x,y):   
    ''' function to find the correlation of two arrays'''
    xAvg=Avg(x)
    yAvg=Avg(y)
    rxy=0.
    n=min(len(x),len(y))
    for k in range(n):
        rxy=rxy+(x[k]-xAvg)*(y[k]-yAvg)
    rxy=rxy/(k+1)
    stdDevx=stdDev(x)
    stdDevy=stdDev(y)
    rxy=rxy/(stdDevx*stdDevy)
    return rxy

###############################################
# Variables
###############################################
TorP='p'

if TorP=='p':
    precip=True
    temp=False
if TorP=='t':
    temp=True
    precip=False
years=65

if precip:
    cityFile='precip_cities_over_'+str(years)+'_years_no_usc'
    var=('Total_Yearly_Precip','Precip_from_Very_Wet_Days','Max_Number_of_Consecutive_Dry_Days',
        'Precip_From_Wettest_Consecutive_Five_Days','Wettest_Day_of_Year')
    units=('total yearly precip change, inches per century',
        'yearly precip change, inches per century','max # of days, change per century',
        'yearly precip change, inches per century','change in inches per century')
    titleVar=('TOTAL YEARLY PRECIPITATION','WETTEST DAYS: Precip from Days >95th Percentile',
        'DRY SPELLS: Max Number of Consecutive Dry Days',
        'WET SPELLS: Precip from Wettest Consecutive Five Days Each Year',
        'Wettest Day of Year')
    flipCM=(True, True,False,True,True)
    binSize=(1,1,2,.2,.1)
    ext=(25,20,30,3,2)
    interval=(5,5,5,1,1)
    colorbarExt=(18,10,10,2,1)
    v=2
    
if temp:
    cityFile='tmax_cities_over_'+str(years)+'_years_no_usc'
    #cityFile='check_cities1'
    var=('Yearly_Tmax_Avg','Yearly_Tmin_Avg','Warm_Days','Warm_Nights','Cold_Days','Cold_Nights',
        'Tmax_Heat_Waves','Tmin_Heat_Waves','Tmax_Cold_Spells','Tmin_Cold_Spells',
        'Warmest_Tmax_Day','Warmest_Tmin_Day','Coldest_Tmax_Day','Coldest_Tmin_Day',
        'Frost_Nights','Tropical_Nights')
    titleVar=('YEARLY AVERAGE of Daily High Temperature (Tmax)', #0
        'YEARLY AVERAGE of Daily Low Temperature (Tmin)',#1
        'WARM DAYS: Number of Daily Highs >90th Percentile', #2
        'WARM NIGHTS: Number of Daily Lows >90th Percentile', #3
        'COLD DAYS: Number of Daily Highs <10th Percentile', #4
        'COLD NIGHTS: Number of Daily Lows <10th Percentile', #5
        'HEAT WAVES: 3 Consecutive Days with Highs >90th Percentile', #6
        'HEAT WAVES: 3 Consecutive Days with Lows >90th Percentile', #7
        'COLD SPELLS: 3 Consecutive Days with Highs <10th Percentile', #8
        'COLD SPELLS: 3 Consecutive Days with Lows <10th Percentile', #9
        'WARMEST DAY: Warmest High Temperature of the Year', #10
        'WARMEST NIGHT: Warmest Low Temperature of the Year', #11
        'COLDEST DAY: Lowest High Temperature of the Year', #12
        'COLDEST NIGHT: Lowest Low Temperature of the Year', #13
        'FROST NIGHTS: Number of Nights Below 32 F (0 C)', #14
        'TROPICAL NIGHTS: Nights that Stay Above 68 F (20 C)') #15
    units=('degrees F per century','degrees F per century','# of days per year, change per century',
        '# of days per year, change per century','# of days per year, change per century',
        '# of days per year, change per century','# per year, change per century',
        '# per year, change per century','# per year, change per century',
        '# per year, change per century','degrees F per century',
        'degrees F per century','degrees F per century','degrees F per century',
        '# of days per year, change per century','# of days per year, change per century')
    flipCM=(True,True,True,True,False,False,
        True,True,False,False,
        True,True,True,True,
        False,True)
    binSize=(0.5,0.5,5,5,5,5,1,1,1,1,1,1,1,1,5,1)
    ext=(10,10,50,50,50,50,16,16,16,16,10,10,18,18,70,10)
    interval=(2,2,10,10,10,10,2,2,2,2,2,2,2,2,10,2)
    colorbarExt=(5,5,40,50,30,30,10,12,9,14,8,8,10,10,50,10)
    v=0
    
# Tmax iplot=0
# Tmin iplot=1
# Warm days iplot=2
# Warm nights iplot=3
# Cold days iplot=4
# Cold nights iplot=5
# Heat waves of highs=6
# Heat waves of lows=7
# Cold Spells of highs=8
# Cold Spells of lows=9
# Warmest Daily High=10
# Warmest Daily Low=11
# Coldest Daily High=12
# Coldest Daily Low=13
# Frost Nights=14
# Tropical Nights=15


#              #
# read in data #
#              # 
slope=pickle.load(open('final_data/'+cityFile+'/slope.p','rb'))
R_squared=pickle.load(open('final_data/'+cityFile+'/R2.p','rb'))
latAll=pickle.load(open('final_data/'+cityFile+'/lat.p','rb'))
lonAll=pickle.load(open('final_data/'+cityFile+'/lon.p','rb'))
station=pickle.load(open('final_data/'+cityFile+'/station.p','rb'))
goodCity=pickle.load(open('final_data/'+cityFile+'/goodCity.p','rb'))
cityList=pickle.load(open('final_data/'+cityFile+'/cityList.p','rb'))
lat=latAll[goodCity[v,:]]
lon=lonAll[goodCity[v,:]]
numGoodCities=sum(goodCity[v,:])

badCityCold=0
badCityHeat=0

###########################################################################
# Fahrenheit
###########################################################################

if temp:
    ###############################################
    # CSV file for Means
    ###############################################
    CSVMeans = open('CSVfiles/Temperature_Means.csv', 'w')
    
    #Change in yearly average of daily lows: deg F/century
    CSVMeans.write('%s,' % 'Station')
    CSVMeans.write('%s,' % 'Link to Plots')
    CSVMeans.write('%s,' % 'Change in Daily Highs: F/century')
    CSVMeans.write('%s,' % 'Change in Daily Lows: F/century')
    CSVMeans.write('%s,' % 'Change in Daily Highs: C/century')
    CSVMeans.write('%s,' % 'Change in Daily Lows: C/century')
    CSVMeans.write('%s,' % 'Station ID')
    CSVMeans.write('%s,' % 'Latitude')
    CSVMeans.write('%s,' % 'Longitude')
    CSVMeans.write('\n')
    
    for icity in range(len(latAll)):
        if goodCity[v,icity]==False: # don't do bad cities
            continue
            
        city=cityList[icity].title()
        
        CSVMeans.write('%s,' % city)
        CSVMeans.write('%s,' % 'http://lillianpetersen.neocities.org/Tmean')
        CSVMeans.write('%3.2f,' % slope[0,icity])
        CSVMeans.write('%3.2f,' % slope[1,icity])
        CSVMeans.write('%s,' % str(round((slope[0,icity])*5./9.,2)))
        CSVMeans.write('%s,' % str(round((slope[1,icity])*5./9.,2)))
        CSVMeans.write('%s,' % station[icity])
        CSVMeans.write('%3.2f,' % latAll[icity])
        CSVMeans.write('%3.2f,' % lonAll[icity])
        CSVMeans.write('\n')  
    CSVMeans.close()
    
    ###############################################
    # CSV file for Heat Extremes
    ###############################################
    CSVHeatExtremes = open('CSVfiles/Heat_Extremes.csv', 'w')
    
    CSVHeatExtremes.write('%s,' % 'Station')
    CSVHeatExtremes.write('%s,' % 'Link to Plots')
    CSVHeatExtremes.write('%s,' % 'Warm Days Change: #/yr/century')
    CSVHeatExtremes.write('%s,' % 'Heat Waves Change: #/yr/century')
    CSVHeatExtremes.write('%s,' % 'Warmest Day Change: F/century')
    CSVHeatExtremes.write('%s,' % 'Station ID')
    CSVHeatExtremes.write('%s,' % 'Latitude')
    CSVHeatExtremes.write('%s,' % 'Longitude')
    CSVHeatExtremes.write('\n')
    
    for icity in range(len(latAll)):
        if goodCity[v,icity]==False: # don't do bad cities
            badCityHeat+=1
            continue
            
        city=cityList[icity].title()
        
        CSVHeatExtremes.write('%s,' % city)
        CSVHeatExtremes.write('%s,' % 'http://lillianpetersen.neocities.org/T_heat_ext')
        CSVHeatExtremes.write('%3.2f,' % slope[2,icity])
        CSVHeatExtremes.write('%3.2f,' % slope[6,icity])
        CSVHeatExtremes.write('%3.2f,' % slope[10,icity])
        CSVHeatExtremes.write('%s,' % station[icity])
        CSVHeatExtremes.write('%3.2f,' % latAll[icity])
        CSVHeatExtremes.write('%3.2f,' % lonAll[icity])
        CSVHeatExtremes.write('\n')  
    CSVHeatExtremes.close()
    
    ###############################################
    # CSV file for Cold Extremes in F
    ###############################################
    CSVColdExtremes = open('CSVfiles/Cold_Extremes.csv', 'w')
    
    CSVColdExtremes.write('%s,' % 'Station')
    CSVColdExtremes.write('%s,' % 'Link to Plots')
    CSVColdExtremes.write('%s,' % 'Cold Days Change: #/yr/century')
    CSVColdExtremes.write('%s,' % 'Cold Spells Change: #/yr/century')
    CSVColdExtremes.write('%s,' % 'Frost Nights Change: #/yr/century')
    CSVColdExtremes.write('%s,' % 'Station ID')
    CSVColdExtremes.write('%s,' % 'Latitude')
    CSVColdExtremes.write('%s,' % 'Longitude')
    CSVColdExtremes.write('\n')
    
    for icity in range(len(latAll)):
        if goodCity[v,icity]==False: # don't do bad cities
            badCityCold+=1
            continue
            
        city=cityList[icity].title()
        
        CSVColdExtremes.write('%s,' % city)
        CSVColdExtremes.write('%s,' % 'http://lillianpetersen.neocities.org/T_cold_ext')
        CSVColdExtremes.write('%3.2f,' % slope[5,icity])
        CSVColdExtremes.write('%3.2f,' % slope[9,icity])
        CSVColdExtremes.write('%3.2f,' % slope[14,icity])
        CSVColdExtremes.write('%s,' % station[icity])
        CSVColdExtremes.write('%3.2f,' % latAll[icity])
        CSVColdExtremes.write('%3.2f,' % lonAll[icity])
        CSVColdExtremes.write('\n')  
    CSVColdExtremes.close()
    

if precip:
    ###############################################
    # CSV file for Precipitation Means
    ###############################################
    CSVPrecip = open('CSVfiles/Precip_Means.csv', 'w')
    
    #Change in yearly average of daily lows: deg F/century
    CSVPrecip.write('%s,' % 'Station')
    CSVPrecip.write('%s,' % 'Link to Plots')
    CSVPrecip.write('%s,' % 'Precipitation Change: inches')
    CSVPrecip.write('%s,' % 'Precipitation Change: cm')
    CSVPrecip.write('%s,' % 'Station ID')
    CSVPrecip.write('%s,' % 'Latitude')
    CSVPrecip.write('%s,' % 'Longitude')
    CSVPrecip.write('\n')
    
    for icity in range(len(latAll)):
        if goodCity[v,icity]==False: # don't do bad cities
            continue
            
        city=cityList[icity].title()
        
        CSVPrecip.write('%s,' % city)
        CSVPrecip.write('%s,' % 'http://lillianpetersen.neocities.org/PrecipMeans')
        CSVPrecip.write('%3.2f,' % slope[0,icity])
        CSVPrecip.write('%s,' % str(round(slope[0,icity]/0.3937)))
        CSVPrecip.write('%s,' % station[icity])
        CSVPrecip.write('%3.2f,' % latAll[icity])
        CSVPrecip.write('%3.2f,' % lonAll[icity])
        CSVPrecip.write('\n')  
    CSVPrecip.close()
    
    
    ###############################################
    # CSV file for Precipitation Means
    ###############################################
    CSVPrecipExt = open('CSVfiles/Precip_Extremes.csv', 'w')
    
    #Change in yearly average of daily lows: deg F/century
    CSVPrecipExt.write('%s,' % 'Station')
    CSVPrecipExt.write('%s,' % 'Link to Plots')
    CSVPrecipExt.write('%s,' % 'Wettest Days')
    CSVPrecipExt.write('%s,' % 'Wet Spells')
    CSVPrecipExt.write('%s,' % 'Dry Spells')
    CSVPrecipExt.write('%s,' % 'Station ID')
    CSVPrecipExt.write('%s,' % 'Latitude')
    CSVPrecipExt.write('%s,' % 'Longitude')
    CSVPrecipExt.write('\n')
    
    for icity in range(len(latAll)):
        if goodCity[v,icity]==False: # don't do bad cities
            continue
            
        city=cityList[icity].title()
        
        CSVPrecipExt.write('%s,' % city)
        CSVPrecipExt.write('%s,' % 'http://lillianpetersen.neocities.org/PrecipExt')
        CSVPrecipExt.write('%3.2f,' % slope[1,icity])
        CSVPrecipExt.write('%3.2f,' % slope[3,icity])
        CSVPrecipExt.write('%3.2f,' % slope[2,icity])
        CSVPrecipExt.write('%s,' % station[icity])
        CSVPrecipExt.write('%3.2f,' % latAll[icity])
        CSVPrecipExt.write('%3.2f,' % lonAll[icity])
        
        CSVPrecipExt.write('\n')  
    CSVPrecipExt.close()
    
    
    
    
    
    
    
    


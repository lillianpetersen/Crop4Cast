#####################################################################################
# Plots different variables for years 1970-80 and 2090-2100: low and high emmisions
#####################################################################################
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

presentGrowingCounties=pickle.load(open('pickle_files/presentGrowingCounties.p','rb'))
BBcounty=pickle.load(open('pickle_files/BBcounty2.p','rb'))
BBsummerPast=pickle.load(open('pickle_files/BBsummerPast.p','rb'))
BBHeatWavesPast=pickle.load(open('pickle_files/BBHeatWavesPast.p','rb'))
BBKDDpast=pickle.load(open('pickle_files/BBKDDpast.p','rb'))
SummerAvgFuture=pickle.load(open('pickle_files/SummerAvgFuture.p','rb'))
KDDFuture=pickle.load(open('pickle_files/KDDFuture.p','rb'))
HeatWavesFuture=pickle.load(open('pickle_files/HeatWavesFuture.p','rb'))

BBsummerFuture85=-9999*ones(shape=(3143,10))
BBHeatWavesFuture85=-9999*ones(shape=(3143,10))
BBKDDfuture85=-9999*ones(shape=(3143,10))
BBsummerFuture45=-9999*ones(shape=(3143,10))
BBHeatWavesFuture45=-9999*ones(shape=(3143,10))
BBKDDfuture45=-9999*ones(shape=(3143,10))
for icity in range(3143):
    if BBcounty[icity]==False:
        continue
    for y in range(10):
        BBsummerFuture85[icity,y]=SummerAvgFuture[0,icity,y+90]
        BBHeatWavesFuture85[icity,y]=HeatWavesFuture[0,icity,y+90]
        BBKDDfuture85[icity,y]=KDDFuture[0,icity,y+90]
        
        BBsummerFuture45[icity,y]=SummerAvgFuture[1,icity,y+90]
        BBHeatWavesFuture45[icity,y]=HeatWavesFuture[1,icity,y+90]
        BBKDDfuture45[icity,y]=KDDFuture[1,icity,y+90]              
       
BBsummerPast=BBsummerPast[:,0:9].ravel()
BBHeatWavesPast=BBHeatWavesPast[:,0:9].ravel()
BBKDDpast=BBKDDpast[:,0:9].ravel()

BBsummerFuture85=BBsummerFuture85.ravel()
BBHeatWavesFuture85=BBHeatWavesFuture85.ravel()
BBKDDfuture85=BBKDDfuture85.ravel()

BBsummerFuture45=BBsummerFuture45.ravel()
BBHeatWavesFuture45=BBHeatWavesFuture45.ravel()
BBKDDfuture45=BBKDDfuture45.ravel()

goodCityFutureLong85Summer=zeros(shape=(62860),dtype=bool)
goodCityFutureLong45Summer=zeros(shape=(62860),dtype=bool)
goodCityPastLongSummer=zeros(shape=(62860),dtype=bool)

goodCityFutureLong85HeatWaves=zeros(shape=(62860),dtype=bool)
goodCityFutureLong45HeatWaves=zeros(shape=(62860),dtype=bool)
goodCityPastLongHeatWaves=zeros(shape=(62860),dtype=bool)

goodCityFutureLong85KDD=zeros(shape=(62860),dtype=bool)
goodCityFutureLong45KDD=zeros(shape=(62860),dtype=bool)
goodCityPastLongKDD=zeros(shape=(62860),dtype=bool)

j=0
for k in range(28287):
    if BBsummerPast[k]>0:
        BBsummerPast[k]=(BBsummerPast[k]-32.)*(5./9.)
        goodCityPastLongSummer[k]=True
        j+=1
    if BBsummerFuture85[k]>0:
        BBsummerFuture85[k]=(BBsummerFuture85[k]-32.)*(5./9.)
        goodCityFutureLong85Summer[k]=True
    if BBsummerFuture45[k]>0:
        BBsummerFuture45[k]=(BBsummerFuture45[k]-32.)*(5./9.)
        goodCityFutureLong45Summer[k]=True
    if BBHeatWavesPast[k]>0:
        goodCityPastLongHeatWaves[k]=True
        j+=1
    if BBHeatWavesFuture85[k]>0:
        goodCityFutureLong85HeatWaves[k]=True
    if BBHeatWavesFuture45[k]>0:
        goodCityFutureLong45HeatWaves[k]=True
    
    if BBKDDpast[k]>0:
        goodCityPastLongKDD[k]=True
        j+=1
    if BBKDDfuture85[k]>0:
        goodCityFutureLong85KDD[k]=True
    if BBKDDfuture45[k]>0:
        goodCityFutureLong45KDD[k]=True

BBvar=(BBsummerPast,BBHeatWavesPast,BBKDDpast,BBsummerFuture85,BBHeatWavesFuture85,BBKDDfuture85,BBsummerFuture45,BBHeatWavesFuture45,BBKDDfuture45)
var=('Summer_Avg_Temp','Heat_Waves','KDD','Summer_Avg_Temp','Heat_Waves','KDD')
labelVar=('Summer Avg Temperature, C','Heat Waves','Killing Degree Days','Summer Avg Temp','Heat Waves','KDD')
titleVar=('Summer Average Temperature for Bread Basket', #0
    'Heat Waves for Bread Basket',#1
    'Killing Degree Days for Bread Basket') 
binsize=(0.25,1,50,0.25,1,50,0.25,1,50)
minext=(24,0,100,24,0,100,24,0,100)
maxext=(42,50,2500,42,50,2500,42,50,2500)
goodCityLong=(goodCityPastLongSummer,goodCityPastLongHeatWaves,goodCityPastLongKDD,
    goodCityFutureLong85Summer,goodCityFutureLong85HeatWaves,goodCityFutureLong85KDD,
    goodCityFutureLong45Summer,goodCityFutureLong45HeatWaves,goodCityFutureLong45KDD)
ymax=(480,1000,750)

if not os.path.exists(r'final_figures/hists_different_times'):
    os.makedirs(r'final_figures/hists_different_times')
    
for k in range(3):
    #figure(1,figsize=(9,5))
    #           #
    # Histogram #
    #           #
    bins1=np.linspace(minext[k],maxext[k],(maxext[k]-minext[k])/binsize[k]+1)
    (mu1, sigma1) = norm.fit(BBvar[k][goodCityLong[k]]) # fit the best fit normal distrobution
    
    bins3=np.linspace(minext[k+6],maxext[k+6],(maxext[k+6]-minext[k+6])/binsize[k+6]+1)
    (mu3, sigma3) = norm.fit(BBvar[k+6][goodCityLong[k+6]]) # fit the best fit normal distrobution
    
    bins2=np.linspace(minext[k+3],maxext[k+3],(maxext[k+3]-minext[k+3])/binsize[k+3]+1)
    (mu2, sigma2) = norm.fit(BBvar[k+3][goodCityLong[k+3]]) # fit the best fit normal distrobution
    
    # plot the histogram
    n, bins, patches = plt.hist(BBvar[k][goodCityLong[k]], bins1, normed=False, histtype='bar', 
        facecolor='green', alpha=0.6, label='1970-1980')
    n, bins, patches = plt.hist(BBvar[k+6][goodCityLong[k+6]], bins3, normed=False, histtype='bar', 
        facecolor='blue', alpha=0.6, label='2090-2100, low emissions')
    n, bins, patches = plt.hist(BBvar[k+3][goodCityLong[k+3]], bins2, normed=False, histtype='bar', 
        facecolor='red', alpha=0.6, label='2090-2100, high emissions')
    plt.legend(loc='upper center')
    if k==0:
        plt.xlabel(labelVar[k])
    else:
        plt.xlabel('Number of '+labelVar[k]+'/year')
    plt.ylabel('Number of instances (years and counties)')
    plt.title(titleVar[k])
    y1 = mlab.normpdf( bins1, mu1, sigma1) # best fit normal ditrobution
    y3 = mlab.normpdf( bins3, mu3, sigma3) # best fit normal ditrobution
    y2 = mlab.normpdf( bins2, mu2, sigma2) # best fit normal ditrobution
    l1 = plt.plot(bins1, y1*binsize[k]*np.size(BBvar[k][goodCityLong[k]]), 'r-', linewidth=2)
    l3 = plt.plot(bins3, y3*binsize[k+6]*np.size(BBvar[k+6][goodCityLong[k+6]]), 'r-', linewidth=2)
    l2 = plt.plot(bins2, y2*binsize[k+3]*np.size(BBvar[k+3][goodCityLong[k+3]]), 'r-', linewidth=2)
    plt.axvline(BBvar[k][goodCityLong[k]].mean(), color='r', linewidth=4) # line at the mean
    plt.axvline(BBvar[k+6][goodCityLong[k+6]].mean(), color='r', linewidth=4) # line at the mean
    plt.axvline(BBvar[k+3][goodCityLong[k+3]].mean(), color='r', linewidth=4) # line at the mean
    axes = plt.gca()
    axes.set_ylim([0,ymax[k]])
    grid(True)
    plt.savefig('final_figures/hists_different_times/'+var[k]+'_overlap_times_C',dpi=500)
    plt.show()
    pause(.5)
    exit()
    clf()

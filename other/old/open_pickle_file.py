################################################
# The Third Code
# Makes the aggregated plots
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
TorP='t'

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
#              #
# read in data #
#              # 
slope=pickle.load(open('final_data/'+cityFile+'/slope.p','rb'))
R_squared=pickle.load(open('final_data/'+cityFile+'/R2.p','rb'))
lat=pickle.load(open('final_data/'+cityFile+'/lat.p','rb'))
lon=pickle.load(open('final_data/'+cityFile+'/lon.p','rb'))
station=pickle.load(open('final_data/'+cityFile+'/station.p','rb'))
goodCity=pickle.load(open('final_data/'+cityFile+'/goodCity.p','rb'))
lat=lat[goodCity[v,:]]
lon=lon[goodCity[v,:]]
numGoodCities=sum(goodCity[v,:])
R2Slopecorr=zeros(shape=(len(var)))

for k in range(len(var)): 
    
    #################################################################
    # Make World Graphs
    #################################################################
    
    R2Slopecorr[k]=corr(slope[k,goodCity[v,:]],R_squared[k,goodCity[v,:]])
    
    for icity in range(len(lat)):
        if goodCity[v,icity]==False: # don't do bad cities
           continue
        latIndex=round(lat[icity])+90
        lonIndex=round(lon[icity])+180
    
    
    #                         #
    # Make world slope graphs #
    #                         #
    if temp:
        cm = plt.cm.get_cmap('RdYlBu') # the colorbar
    if precip:
        cm = plt.cm.get_cmap('BrBG') # the colorbar
        if not flipCM[k]:
            cm=reverse_colormap(cm)
    if temp:
        if flipCM[k]:
            cm = reverse_colormap(cm)
    #figure(1)
    figure(4,figsize=(20,20))
    
    # get the basemap
    m = Basemap(llcrnrlat=-45,urcrnrlat=75,\
            llcrnrlon=-173,urcrnrlon=180,lat_ts=50,resolution='i',area_thresh=10000)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#BDFCC9')
    x,y=m(lon,lat)
    
    slope1d=slope[k,goodCity[v,:]]
    sortedind=np.argsort(slope1d)
    
    if not flipCM[k] and temp:
        sortedind=sortedind[::-1]
    
    if temp:
        # plot the scatter plot
        plotHandle = m.scatter(x[sortedind],y[sortedind],c=slope1d[sortedind],s=50,cmap=cm, 
            edgecolor='none', vmin=-1*colorbarExt[k],vmax=colorbarExt[k],zorder=10)
        m.colorbar(plotHandle)
    if precip:
        # plot the scatter plot
        plotHandle = m.scatter(x,y,c=slope[k,goodCity[v,:]],s=50,cmap=cm, 
            edgecolor='none', vmin=-1*colorbarExt[k],vmax=colorbarExt[k],zorder=10)
        m.colorbar(plotHandle)
        
    title(str(titleVar[k]))
    xlabel('slope: '+units[k])
    if not os.path.exists(r'final_figures/'+cityFile):
            os.makedirs(r'final_figures/'+cityFile)
    savefig('final_figures/'+cityFile+'/'+str(var[k])+'_slope',dpi=500)
    show() 
    exit()
    clf()
    
    #                      #
    # make R2 world graphs #
    #                      #
    figure(3)
    cm = plt.cm.get_cmap('brg') # the colorbar
    cm=reverse_colormap(cm)
    
    # get the basemap
    m = Basemap(llcrnrlat=-45,urcrnrlat=75,\
            llcrnrlon=-173,urcrnrlon=180,lat_ts=50,resolution='c',area_thresh=10000)
    m.drawcoastlines()
    m.fillcontinents(color='#EEE685')
    x,y=m(lon,lat)
    R21d=R_squared[k,goodCity[v,:]]
    sortedind=np.argsort(R21d)
    
    # plot the scatter plot
    plotHandle = m.scatter(x[sortedind],y[sortedind],c=R21d[sortedind],s=8,
        cmap=cm,edgecolor='none', vmin=0,vmax=.5, zorder=10)
    m.colorbar(plotHandle,ticks=[0,.1,.2,.3,.4,.5],)
        #boundaries=[0] + bounds + [.6])
    title('R2 of '+str(titleVar[k]))
    xlabel('R2 of slope, correlation between R2 and slope = '+str(round(R2Slopecorr[k],2)))
    savefig('final_figures/'+cityFile+'/'+str(var[k])+'_R2',dpi=500)
    show()
    clf()
    
    # the significant R2 values are:
    # 1 in 20 chance (significant): .06
    # 1 in 100 chance (highly significant): .1
    # 1 in 1,000 chance: .16
    # 1 in 10,000 chance: .22
    # 1 in 100,000 chance: .27
    # 1 in 1,000,000 chance: .32
    
    
    #                       #
    # make slope histograms #
    #                       #
    f=figure(2,figsize=(7, 3)) # set the size of the figure
    ax = f.add_subplot(111)
    (mu, sigma) = norm.fit(slope[k,goodCity[v,:]]) # fit the best fit normal distrobution
    
    tmp=slope[k,goodCity[v,:]]
    
    cityless0=size(tmp[tmp<0])
    citymore0=size(tmp[tmp>0])
    
    cityless0percent=size(tmp[tmp<0])/float(numGoodCities)*100
    citymore0percent=size(tmp[tmp>0])/float(numGoodCities)*100
    
    bin_size = binSize[k]; max_edge = 100; min_edge = -1*max_edge 
    N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
    bin_list = np.linspace(min_edge, max_edge, Nplus1)
    
    # plot the histogram
    n, bins, patches = plt.hist(slope[k,goodCity[v,:]], bins=bin_list,
        facecolor='blue',alpha=0.75)
    plt.xlabel('slope: '+units[k])
    plt.ylabel('Number of Stations')
    plt.title(str(titleVar[k]))
    
    # add text
    text(0.02, 0.875,str(cityless0)+' stations\n'+str(round(cityless0percent,1))+'%',
        horizontalalignment='left',
        verticalalignment='center',
        transform = ax.transAxes)
    subplots_adjust(bottom=0.17)
    
    # add text
    text(0.98, 0.875,str(citymore0)+' stations\n'+str(round(citymore0percent,1))+'%',
        horizontalalignment='right',
        verticalalignment='center',
        transform = ax.transAxes)
    subplots_adjust(bottom=0.17)
    #plt.axis([-.1,.1,0,5])
    plt.grid(True)
    
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y*numGoodCities*binSize[k], 'r-', linewidth=2)
    
    plt.axvline(slope[k,goodCity[v,:]].mean(), color='r', linewidth=4) # line at the mean
    plt.axvline(0, color='k', linewidth=3) # line at 0

    xticks(range(-1*ext[k],ext[k]+1,interval[k]))
    xlim(-1*ext[k],ext[k])
    markeredgewidth=0.0
    plt.savefig('final_figures/'+cityFile+'/'+str(var[k])+'_slope_histogram',dpi=500)
    plt.show()
    plt.clf()
    
    
    #################################################################
    # Make Zoomed in Graphs
    #################################################################
    
    ##             ##
    ## Zoom for US ##
    ##             ##
    if temp:
        cm = plt.cm.get_cmap('RdYlBu') # the colorbar
    if precip:
        cm = plt.cm.get_cmap('BrBG') # the colorbat
        if not flipCM[k]:
            cm=reverse_colormap(cm)
    if temp:
        if flipCM[k]:
            cm = reverse_colormap(cm)
    #figure(4,figsize=(20,20)) # choose the size of the figure
    figure(5,figsize=(9,7))
    show()
    
    # get the basemap
    m = Basemap(llcrnrlat=23,urcrnrlat=55,\
            llcrnrlon=-128,urcrnrlon=-62,lat_ts=50,resolution='i',area_thresh=10000)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries(linewidth=2)
    m.fillcontinents(color='#BDFCC9')
    x,y=m(lon,lat)
    
    slope1d=slope[k,goodCity[v,:]]
    sortedind=np.argsort(slope1d)
    
    if not flipCM[k] and temp:
        sortedind=sortedind[::-1]
        
    if years==65:
        dotsize=150
    if years==100:
        dotsize=180
        
    dotsize=50
    
    if temp:
        # plot the zommed figure
        plotHandle = m.scatter(x[sortedind],y[sortedind],c=slope1d[sortedind],s=dotsize,cmap=cm, 
            edgecolor='none', vmin=-1*colorbarExt[k],vmax=colorbarExt[k],zorder=10)
        m.colorbar(plotHandle)
    if precip:
        # plot the zoomed figure
        plotHandle = m.scatter(x,y,c=slope[k,goodCity[v,:]],s=dotsize,cmap=cm, 
            edgecolor='k', vmin=-1*colorbarExt[k],vmax=colorbarExt[k],zorder=10)
        m.colorbar(plotHandle)
        
    title(str(titleVar[k])+' for cities over '+str(years)+' years')
    xlabel('slope: '+units[k])
    if not os.path.exists(r'final_figures/'+cityFile):
            os.makedirs(r'final_figures/'+cityFile)
    if not os.path.exists(r'final_figures/'+cityFile+'/US_zoom'):
            os.makedirs(r'final_figures/'+cityFile+'/US_zoom')
    savefig('final_figures/'+cityFile+'/US_zoom/'+str(var[k])+'_slope_US_zoom_small',dpi=500)
    show() 
    clf()
    
    dotsize=50
    ##                     ##
    ## Zoom for Austrailia ##
    ##                     ##
    if temp:
        cm = plt.cm.get_cmap('RdYlBu') # the colorbar
    if precip:
        cm = plt.cm.get_cmap('BrBG') # the colorbar
        if not flipCM[k]:
            cm=reverse_colormap(cm)
    if temp:
        if flipCM[k]:
            cm = reverse_colormap(cm)
    #figure(4,figsize=(20,20)) # choose the size of the figure
    figure(5,figsize=(9,7))
    
    # get the basemap
    m = Basemap(llcrnrlat=-40,urcrnrlat=-10,\
            llcrnrlon=112,urcrnrlon=154,lat_ts=50,resolution='i',area_thresh=10000)
    m.drawcoastlines()
    m.drawstates()
    m.fillcontinents(color='#BDFCC9')
    x,y=m(lon,lat)
    
    slope1d=slope[k,goodCity[v,:]]
    sortedind=np.argsort(slope1d)
    
    if not flipCM[k] and temp:
        sortedind=sortedind[::-1]
    if precip:
        sortedind=sortedind[::-1]
    
    if temp:
        # plot the figure
        plotHandle = m.scatter(x[sortedind],y[sortedind],c=slope1d[sortedind],s=dotsize,cmap=cm, 
            edgecolor='none', vmin=-1*colorbarExt[k],vmax=colorbarExt[k],zorder=10)
        m.colorbar(plotHandle)
    if precip:
        # plot the figure
        plotHandle = m.scatter(x[sortedind],y[sortedind],c=slope1d[sortedind],s=70,cmap=cm, 
            edgecolor='none', vmin=-1*colorbarExt[k],vmax=colorbarExt[k],zorder=10)
        m.colorbar(plotHandle)
        
    title(str(titleVar[k])+' for cities over '+str(years)+' years')
    xlabel('slope: '+units[k])
    if not os.path.exists(r'final_figures/'+cityFile):
            os.makedirs(r'final_figures/'+cityFile)
    if not os.path.exists(r'final_figures/'+cityFile+'/Australia_zoom'):
            os.makedirs(r'final_figures/'+cityFile+'/Australia_zoom')
    savefig('final_figures/'+cityFile+'/Australia_zoom/'+str(var[k])+'_slope_Australia_zoom',dpi=500)
    show() 
    clf()
    
    ##                        ##
    ## Zoom for Austrailia R2 ##
    ##                        ##
    cm = plt.cm.get_cmap('brg')
    cm=reverse_colormap(cm)
    #figure(4,figsize=(20,20)) # choose the size of the figure
    figure(5,figsize=(9,7))
    show()
    # get the basemap
    m = Basemap(llcrnrlat=-40,urcrnrlat=-10,\
            llcrnrlon=112,urcrnrlon=154,lat_ts=50,resolution='i',area_thresh=10000)
    m.drawcoastlines()
    m.drawstates()
    m.fillcontinents(color='#EEE685')
    x,y=m(lon,lat)
    R21d=R_squared[k,goodCity[v,:]]
    sortedind=np.argsort(R21d)
    
    # plot the figure
    plotHandle = m.scatter(x[sortedind],y[sortedind],c=R21d[sortedind],s=50,
        cmap=cm,edgecolor='none', vmin=0,vmax=.5, zorder=10)
    m.colorbar(plotHandle,ticks=[0,.1,.2,.3,.4,.5])
    title('R2 of '+str(titleVar[k]))
    xlabel('R2 of slope, correlation between R2 and slope = '+str(round(R2Slopecorr[k],2)))
    if not os.path.exists(r'final_figures/'+cityFile+'/Australia_zoom_R2'):
            os.makedirs(r'final_figures/'+cityFile+'/Australia_zoom_R2')
    savefig('final_figures/'+cityFile+'/Australia_zoom_R2/'+str(var[k])+'_R2_Australia_zoom',dpi=500)
    show()
    clf()
    
    
    ##                 ##
    ## Zoom for Europe ##
    ##                 ##
    if temp:
        cm = plt.cm.get_cmap('RdYlBu') # the colorbar
    if precip:
        cm = plt.cm.get_cmap('BrBG') # the colorbar
        if not flipCM[k]:
            cm=reverse_colormap(cm)
    if temp:
        if flipCM[k]:
            cm = reverse_colormap(cm)
    #figure(4,figsize=(20,20)) # choose the size of the figure
    figure(5,figsize=(9,7))
    
    # get the basemap
    m = Basemap(llcrnrlat=25,urcrnrlat=75,\
            llcrnrlon=-15,urcrnrlon=90,lat_ts=50,resolution='i',area_thresh=10000)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#BDFCC9')
    x,y=m(lon,lat)
    
    slope1d=slope[k,goodCity[v,:]]
    sortedind=np.argsort(slope1d)
    
    if not flipCM[k] and temp:
        sortedind=sortedind[::-1]
    
    dotsize=50
    if temp:
        plotHandle = m.scatter(x[sortedind],y[sortedind],c=slope1d[sortedind],s=dotsize,cmap=cm, 
            edgecolor='none', vmin=-1*colorbarExt[k],vmax=colorbarExt[k],zorder=10)
        m.colorbar(plotHandle)
    if precip:
        plotHandle = m.scatter(x,y,c=slope[k,goodCity[v,:]],s=dotsize,cmap=cm, 
            edgecolor='none', vmin=-1*colorbarExt[k],vmax=colorbarExt[k],zorder=10)
        m.colorbar(plotHandle)
        
    title(str(titleVar[k])+' for cities over '+str(years)+' years')
    xlabel('slope: '+units[k])
    if not os.path.exists(r'final_figures/'+cityFile):
            os.makedirs(r'final_figures/'+cityFile)
    if not os.path.exists(r'final_figures/'+cityFile+'/Europe_zoom'):
            os.makedirs(r'final_figures/'+cityFile+'/Europe_zoom')
    savefig('final_figures/'+cityFile+'/Europe_zoom/'+str(var[k])+'_slope_Europe_zoom_small',dpi=500)
    show()
    clf()
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
import random

# cd Documents/Science_Fair_2017_Crop_Yields/

f=open('data/growing_season_corn.csv','r')

startGrowingMon=-9999*ones(shape=(57,3))
endGrowingMon=-9999*ones(shape=(57,3))
startGrowingDay=-9999*ones(shape=(57,3))
endGrowingDay=-9999*ones(shape=(57,3))

for cp in range(3):
    for character in f:
        tmp=character[0:19]
        
        s=int(tmp[3:5])
        startDate=tmp[6:12]
        endDate=tmp[13:19]
        
        if startDate[0:3]=='Apr':
            startMon=3
        if startDate[0:3]=='May':
            startMon=4
        if startDate[0:3]=='Jun':
            startMon=5
        if startDate[0:3]=='Jul':
            startMon=6
        
        if endDate[0:3]=='Aug':
            endMon=7
        if endDate[0:3]=='Sep':
            endMon=8
        if endDate[0:3]=='Oct':
            endMon=9
        if endDate[0:3]=='Nov':
            endMon=10
        
        startGrowingMon[s,cp]=int(startMon)
        startGrowingDay[s,cp]=int(startDate[4:6])
        endGrowingMon[s,cp]=int(endMon)
        endGrowingDay[s,cp]=int(endDate[4:6])
    if cp==0:
        f=open('data/growing_season_soy.csv','r')
    if cp==1:
        f=open('data/growing_season_rice.csv','r')
    
pickle.dump(startGrowingMon,open('pickle_files/startGrowingMon.p','wb'))
pickle.dump(startGrowingDay,open('pickle_files/startGrowingDay.p','wb'))
pickle.dump(endGrowingMon,open('pickle_files/endGrowingMon.p','wb'))
pickle.dump(endGrowingDay,open('pickle_files/endGrowingDay.p','wb'))
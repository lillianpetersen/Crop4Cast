################################################################
# Make pie charts of the cost breakdown
################################################################

import csv
from math import sqrt
from sys import exit
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import os
# from scipy.stats import norm
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import shapefile
from math import sin, cos, sqrt, atan2, radians, pi, degrees
from scipy import ndimage
from matplotlib.font_manager import FontProperties
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import itertools


################################################
# Functions
################################################
def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap
################################################

try:
    wddata='/Users/lilllianpetersen/iiasa/data/supply_chain/'
    wdfigs='/Users/lilllianpetersen/iiasa/figs/supply_chain/'
    wdvars='/Users/lilllianpetersen/iiasa/saved_vars/supply_chain/'
    f=open(wddata+'population/CAPITALVERSIONcasenumbers.csv','r')
except:
    wddata='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/data/'
    wdfigs='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/figs/'
    wdvars='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/vars/'

cVarNames = ['Ingredient','Running','Packaging','Transport']
currentCosts = np.array([43.13,41.56,10.21,5.06])
colors = ['darkorange','royalblue','yellow','limegreen']
explode = [0.01,0.01,0.01,0.01]

#plt.pie(currentCosts, labels=cVarNames, colors=colors)

plt.clf()
fig1, ax1 = plt.subplots()
ax1.pie(currentCosts, labels=cVarNames, explode = explode, autopct='%1.1f%%', shadow=True, startangle=180)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Procurement Costs of the Current Supply Chain',fontsize = 16)
plt.savefig(wdfigs+'current_piechart.pdf')


oVarNames = ['Ingredient','Running','Packaging','Transport','Startup']
optimizedCosts = np.array([40.97, 39.14, 9.62, 9.03, 1.24])
colors = ['darkorange','royalblue','yellow','limegreen','mediumorchid']
explode = [0.01,0.01,0.01,0.01,0.01]


plt.clf()
fig1, ax1 = plt.subplots()
ax1.pie(optimizedCosts, labels=oVarNames, explode = explode, autopct='%1.1f%%', shadow=True, startangle=180)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Procurement Costs of the Optimized Supply Chain',fontsize = 16)
plt.savefig(wdfigs+'optimized_piechart.pdf')





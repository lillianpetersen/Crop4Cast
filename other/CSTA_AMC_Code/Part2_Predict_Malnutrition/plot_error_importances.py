#######################################################
# Step 1: Forecast SNF Demand
# Reads in and interpolates training features
# Trains a random forest regression between training features and mal prevalence
# A train-test split validates the model
# Predicts mal prevalence to 2021
#######################################################

from __future__ import division
import csv
from math import sqrt
from sys import exit
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
#from mpl_toolkits.basemap import Basemap
import os
from scipy.stats import norm, mode
import matplotlib as mpl
from matplotlib.patches import Polygon
import random
import matplotlib.cm as cm
import matplotlib.colors as colors
from PIL import Image
from osgeo import gdal
#import ogr
from IPython import embed
import shapefile
#from shapely.geometry import shape, Point
import matplotlib.patches as patches
from math import sin, cos, sqrt, atan2, radians, pi, degrees
from geopy.geocoders import Nominatim
geolocator = Nominatim()
import geopy.distance
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.interpolate import RectSphereBivariateSpline
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
#from sknn.mlp import Regressor
import sklearn
#import googlemaps

###############################################
# Functions
###############################################
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
	stdDevx=np.std(x)
	stdDevy=np.std(y)
	rxy=rxy/(stdDevx*stdDevy)
	return rxy

def scale(var):
	varScaled=(var-np.amin(var))/(np.amax(var)-np.amin(var))
	return varScaled

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
#, (50,205,50)(173,255,47) , 
#colors = [(0,128,0) , (50,205,50) , (173,255,47) , (255,255,0) , (255,179,25) , (255,69,0) , (139,0,0)]
colors = [(255,255,255) , (50,205,50) ,  (255,255,0) ,(255,213,0) , (255,179,25) ,  (255,69,0) , (255,0,0) , (139,0,0), (0,0,0)]
my_cmap = make_cmap(colors,bit=True)

################################################
try:
	wddata='/Users/lilllianpetersen/iiasa/data/'
	wdfigs='/Users/lilllianpetersen/iiasa/figs/'
	wdvars='/Users/lilllianpetersen/iiasa/saved_vars/'
except:
	wddata='C:/Users/garyk/Documents/python_code/riskAssessmentFromPovertyEstimations/data/'
	wdfigs='C:/Users/garyk/Documents/python_code/riskAssessmentFromPovertyEstimations/figs/'
	wdvars='C:/Users/garyk/Documents/python_code/riskAssessmentFromPovertyEstimations/vars/'

MakePlots=False

latsubsaharan=np.load(wdvars+'latsubsaharan.npy')
lonsubsaharan=np.load(wdvars+'lonsubsaharan.npy')
africaMask1=np.load(wdvars+'africaMasksubsaharan.npy')

f=open(wddata+'boundaries/africanCountries.csv','r')
africanCountries=[]
for line in f:
	africanCountries.append(line[:-1])

#northernCountries = ['Benin', 'Burkina Faso', 'Cameroon', 'Central African Republic', 'Chad', "Cote d'Ivoire", 'Congo (DRC)', 'Djibouti', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Congo', 'Rwanda', 'Senegal', 'Sierra Leone', 'Somalia', 'Sudan', 'South Sudan', 'Togo', 'Uganda', ]
#northernCountries = ['Benin', 'Burkina Faso', 'Cameroon', 'Central African Republic', 'Chad', "Cote d'Ivoire", 'Congo (DRC)', 'Djibouti', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Congo', 'Senegal', 'Sierra Leone', 'Somalia', 'Sudan', 'South Sudan', 'Togo', 'Uganda', ]
northernCountries = np.array(['Benin', 'Burkina Faso', 'Cameroon', 'Central African Republic', 'Chad', "Cote d'Ivoire", 'Congo_(DRC)', 'Djibouti', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 'Somalia', 'Sudan', 'South Sudan', 'Togo', 'Uganda', ])
southernCountries = np.array(['Angola','Rwanda','Burundi','Tanzania','Congo (DRC)','Congo','Zambia','Malawi','Mozambique','Namibia','Botswana','Zimbabwe','South_Africa', 'Lesotho','Swaziland'])
for icountry in range(len(northernCountries)):
	northernCountries[icountry] = northernCountries[icountry].replace(' ','_')
for icountry in range(len(africanCountries)):
	africanCountries[icountry] = africanCountries[icountry].replace(' ','_')


### North
for icountry in range(len(northernCountries)):
	vars()['Importances'+northernCountries[icountry]] = {}
	f = open(wdfigs+'randomForest/'+northernCountries[icountry]+'/importances.csv')
	l=-1
	for line in f:
		l+=1
		tmp = line.split(',')
		if l==0:
			Vars = np.array(tmp)
			Vars[Vars=='bare\n'] = 'bare'
			Vars[Vars=='muslimGrid\n'] = 'muslimGrid'
		else:
			for v in range(len(Vars)):
				vars()['Importances'+northernCountries[icountry]][Vars[v]] = float(tmp[v])

importancesN = np.zeros(shape=(19))
importanceVarsN = Vars
for v in range(len(Vars)):
	Sum = 0
	for icountry in range(len(northernCountries)):
		Sum += vars()['Importances'+northernCountries[icountry]][Vars[v]]
	Avg = Sum/len(northernCountries)
	importancesN[v] = Avg
sortIndex = importancesN.argsort()
importancesN = importancesN[sortIndex][::-1]
importanceVarsN = importanceVarsN[sortIndex][::-1]

VarsNames = ['Precipitation','Open Defecation','War','Elevation','Female Ed','Dist to Coasts','Forest Cover','GPD','Nutrition','Population','Refugees','HDI','Conflicts','Female Ed','Electricity','Christian','Muslim','Agriculture GDP','Bare']

##### Cumulative Importances #####
cumulative_importances = np.cumsum(importancesN)

x=np.arange(len(importancesN))

fig = plt.figure(figsize=(6, 7))
plt.clf()
plt.plot(x,cumulative_importances,'g*-')
plt.hlines(y = 0.97, xmin=0, xmax=len(x), color = 'r', linestyles = 'dashed')
plt.title('Cumalative Importances: Northern Africa')
plt.grid(True,linestyle=':')
plt.xticks(x,VarsNames,rotation='vertical')
plt.gcf().subplots_adjust(bottom=0.2)
plt.ylim([0,1.05])
plt.savefig(wdfigs+'cumulative_importances_north.pdf')

### South
for icountry in range(len(southernCountries)):
	vars()['Importances'+southernCountries[icountry]] = {}
	f = open(wdfigs+'randomForest/'+southernCountries[icountry]+'/importances.csv')
	l=-1
	for line in f:
		l+=1
		tmp = line.split(',')
		if l==0:
			Vars = np.array(tmp)
			Vars[Vars=='bare\n'] = 'bare'
			Vars[Vars=='muslimGrid\n'] = 'muslimGrid'
			Vars[Vars=='warGrid\n'] = 'warGrid'
			Vars[Vars=='refugeesSumGrid\n'] = 'refugeesSumGrid'
		else:
			for v in range(len(Vars)):
				vars()['Importances'+southernCountries[icountry]][Vars[v]] = float(tmp[v])

importancesS = np.zeros(shape=(19))
importanceVarsS = Vars
for v in range(len(Vars)):
	Sum = 0
	for icountry in range(len(southernCountries)):
		Sum += vars()['Importances'+southernCountries[icountry]][Vars[v]]
	Avg = Sum/len(southernCountries)
	importancesS[v] = Avg
sortIndex = importancesS.argsort()
importancesS = importancesS[sortIndex][::-1]
importanceVarsS = importanceVarsS[sortIndex][::-1]

VarsNames = ['Christian','HDI','Forest','Dist to Coasts','Open Defecation','Refugees','War','Precipitation','Population','Female Ed','Elevation','MPconflicts','Muslim','Agriculture GDP','Girl Ed','GPD','Nutrition','Electricity','Bare']

##### Cumulative Importances #####
cumulative_importances = np.cumsum(importancesS)

x=np.arange(len(importancesS))

fig = plt.figure(figsize=(6, 7))
plt.clf()
plt.plot(x,cumulative_importances,'g*-')
plt.hlines(y = 0.97, xmin=0, xmax=len(x), color = 'r', linestyles = 'dashed')
plt.title('Cumalative Importances: Southern Africa')
plt.grid(True,linestyle=':')
plt.xticks(x,VarsNames,rotation='vertical')
plt.gcf().subplots_adjust(bottom=0.2)
plt.ylim([0,1.05])
plt.savefig(wdfigs+'cumulative_importances_south.pdf')

### Boxes
#Importances = {}
#f = open(wdfigs+'boxes_forest/importances.csv')
#l=-1
#for line in f:
#	l+=1
#	tmp = line.split(',')
#	if l==0:
#		Vars = np.array(tmp)
#		Vars[Vars=='bare\n'] = 'bare'
#		Vars[Vars=='muslimGrid\n'] = 'muslimGrid'
#	else:
#		for v in range(len(Vars)):
#			Importances[Vars[v]] = float(tmp[v])
#
#importances = np.zeros(shape=(29))
#for v in range(len(Vars)):
#	importances[v] += Importances[Vars[v]]
#
#sortIndex = importances.argsort()
#importancesN = importances[sortIndex][::-1]
#importanceVarsN = importanceVars[sortIndex][::-1]
#
#VarsNames = ['Precipitation','Open Defecation','War','Elevation','Female Ed','Dist to Coasts','Forest Cover','GPD','Nutrition','Population','Refugees','HDI','Conflicts','Female Ed','Electricity','Christian','Muslim','Agriculture GDP','Bare']

##### Cumulative Importances #####
cumulative_importances = np.cumsum(importancesN)

x=np.arange(len(importancesN))

fig = plt.figure(figsize=(6, 7))
plt.clf()
plt.plot(x,cumulative_importances,'g*-')
plt.hlines(y = 0.97, xmin=0, xmax=len(x), color = 'r', linestyles = 'dashed')
plt.title('Cumalative Importances: Northern Africa')
plt.grid(True,linestyle=':')
plt.xticks(x,VarsNames,rotation='vertical')
plt.gcf().subplots_adjust(bottom=0.2)
plt.ylim([0,1.05])
plt.savefig(wdfigs+'cumulative_importances_north.pdf')



######################################################
# Errors
######################################################

# Corr, np.mean(Error), np.mean(Difference)

error = np.zeros(shape=(len(africanCountries)))
difference = np.zeros(shape=(len(africanCountries)))
corr = np.zeros(shape=(len(africanCountries)))

for icountry in range(len(africanCountries)):
	f = open(wdfigs+'randomForest/'+africanCountries[icountry]+'/error.csv')
	l=-1
	for line in f:
		l+=1
		if l==0:
			corr[icountry] = float(line)
		elif l==1:
			error[icountry] = float(line)
		elif l==2:
			difference[icountry] = float(line)
error = 100*error
difference = 100*difference

errorBoxes = np.load(wdfigs + 'boxes_forest/error.npy')

widths=[0.5,0.5]
fig.clear()
plt.cla()
plt.close()
plt.clf()
plt.figure(25,figsize=(7,3))
ax = plt.subplot(1,1,1)
ax.set_position([.22,.15,.70,.75])
ax.set_aspect(6)
ax.boxplot((error,errorBoxes), 0, '', vert=0, widths=widths, whis=[12.5,87.5])
plt.title('Prediction Errors: Entire Countries')
plt.yticks([1,2],['Countries','Boxes'])
plt.xlabel('Percent Error')
plt.grid(axis='x')
plt.savefig(wdfigs+'box_wisker_error_boxes_and_countries.pdf')

widths=[0.4]
fig.clear()
plt.cla()
plt.close()
plt.clf()
fig = plt.figure(figsize=(7, 3))
ax = plt.subplot(1,1,1)
ax.set_position([.22,.15,.70,.75])
ax.set_aspect(0.13)
plt.boxplot(corr, 0, '', vert=0, widths=widths, whis=[12.5,87.5])
plt.title('Prediction Correlations: Entire Countries')
plt.yticks([1],['Countries'])
plt.xlim([0,1])
plt.xlabel('Correlation')
plt.grid(axis='x')
plt.savefig(wdfigs+'box_wisker_correlation.pdf')

widths=[0.4]
fig.clear()
plt.cla()
plt.close()
plt.clf()
fig = plt.figure(figsize=(7, 3))
ax = plt.subplot(1,1,1)
ax.set_position([.22,.15,.70,.75])
ax.set_aspect(0.8)
ax.set_xlim([1,5])
plt.boxplot(difference, 0, '', vert=0, widths=widths, whis=[12.5,87.5])
plt.title('Prediction Difference: Entire Countries')
plt.yticks([])
plt.xlabel('Difference in Prevalence')
plt.grid(axis='x')
plt.savefig(wdfigs+'box_wisker_difference.pdf')













import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
import sys
from sys import exit
import sklearn
from sklearn import svm
import time
from sklearn.preprocessing import StandardScaler

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'

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

colors = [(.4,0,.6), (0,0,.7), (0,.6,1), (.9,.9,1), (1,.8,.8), (1,1,0), (.8,1,.5), (.1,.7,.1), (.1,.3,.1)]
my_cmap = make_cmap(colors)

#lat=7.51977226909
#lon=37.9036610472
lat=6.439697635729217
lon=36.830086263085065
pixels=1024
nyears=3

ndviClimo=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviClimo.npy')

monthName=('Jan','Feb','March','April','May','June','July','August','Sep','Oct','Nov','Dec')
for m in range(12):
	plt.clf()
	plt.imshow(ndviClimo[m,:,:],vmin=-.4,vmax=.9,cmap=my_cmap)
	plt.colorbar()
	plt.title('ndvi '+monthName[m]+' Climatology Ethiopia')
	plt.savefig(wd+'figures/Ethiopia/climo/ndviClimo_'+monthName[m],dpi=700)

Mask=np.zeros(shape=(12,pixels,pixels))
for m in range(12):
	for v in range(pixels):
		for h in range(pixels):
			if ndviClimo[m,v,h]<0.0 or math.isnan(ndviClimo[m,v,h])==True:
				Mask[m,v,h]=1

climoAvg=np.zeros(shape=(12))
ndviClimoM=np.ma.masked_array(ndviClimo,Mask)
for m in range(12):
	climoAvg[m]=np.ma.mean(ndviClimoM[m,:,:])

climoPlotData=np.zeros(shape=(12))
climoPlotData[:10]=climoAvg[2:]
climoPlotData[10:]=climoAvg[:2]
x=np.arange(3,15)
plt.clf()
plt.plot(x,climoPlotData,'b*-')
plt.xlabel('month')
plt.ylabel('avg monthly ndvi')
plt.title('Average Monthly NDVI for Ethiopia box')
plt.savefig(wd+'figures/Ethiopia/avg_monthly_ndvi_ethiopia',dpi=700)







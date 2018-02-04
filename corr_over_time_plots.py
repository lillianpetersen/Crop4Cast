import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
import sys
from sys import exit
import sklearn
import time
from sklearn.preprocessing import StandardScaler
from operator import and_
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

corrs=np.array([[-0.06,0.12,0.58,0.67], #NDVI
	[-0.09,0.13,0.55,0.67], #EVI
	[0.11,-0.10,-0.58,-0.72]]) #NDWI

multiCorr=np.array([0.8425,0.8425,0.8425,0.8425])

slopes=np.array([[-0.60,0.61,4.0,4.73], #NDVI
	[-0.84,0.49,3.6,3.74], #EVI
	[1.1,-0.48,-3.3,-4.33]]) #NDWI

xint=[5,6,7,8]
x=['May','June','July','August']
plt.clf()
ax=plt.gca()
plt.plot(xint,corrs[1],'^-y',label='EVI')
plt.plot(xint,corrs[0],'s-g',label='NDVI')
plt.plot(xint,-1*corrs[2],'+-b',label='NDWI*(-1)')
plt.plot(xint,multiCorr,'-r',label='Mutivariate Regression')
plt.legend()
plt.ylim(-.20,1.0001)
ax.set_xticks([5,6,7,8])
ax.set_xticklabels(['May','June','July','August'])
plt.title('Correlations Over Time')
plt.ylabel('Correlations to Corn Yield')
plt.grid(True)
plt.savefig(wdfigs+'Illinois/Illinois_corrs_over_time',dpi=700)


plt.clf()
ax=plt.gca()
plt.plot(xint,slopes[1],'^-y',label='EVI')
plt.plot(xint,slopes[0],'s-g',label='NDVI')
plt.plot(xint,-1*slopes[2],'+-b',label='NDWI*(-1)')
plt.legend(loc='lower right')
ax.set_xticks([5,6,7,8])
ax.set_xticklabels(['May','June','July','August'])
plt.title('Slopes Over Time')
plt.ylabel('Crop Yields/(Satellite Index *100)')
plt.grid(True)
plt.savefig(wdfigs+'Illinois/Illinois_slopes_over_time',dpi=700)

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

#corrs=np.array([[-0.06,0.12,0.58,0.67,.24], #NDVI
#	[-0.09,0.13,0.55,0.67,.26], #EVI
#	[0.11,-0.10,-0.58,-0.72,-.32]]) #NDWI
#
#multiCorr=np.array([0.863982,0.863982,0.863982,0.863982,0.863982])
#
#slopes=np.array([[-0.60,0.61,4.0,4.73,1.18], #NDVI
#	[-0.84,0.49,3.6,3.74,.86], #EVI
#	[1.1,-0.48,-3.3,-4.33,-1.66]]) #NDWI

corrs=np.load(wdvars+'Illinois/corr_corn_soy_sorghum.npy') #shape = cp,m,index
multiCorr=np.load(wdvars+'Illinois/corr_multivariate.npy')
slopes=np.load(wdvars+'Illinois/slope_corn_soy_sorghum.npy')

stdDev=np.array([27.366954282941,6.594876031026154,6.251434177106655])
for cp in range(3):
	slopes[cp,:,:]=slopes[cp,:,:]/stdDev[cp]
slopes=slopes*np.mean(stdDev)

plt.clf()
xint=[5,6,7,8,9]
x=['May','June','July','August','September']
ax=plt.gca()
#subPlot1 = plt.axes([.1,.1,.8,.8])
#plt.xticks([])
#plt.yticks([])
#plt.legend()
#
#subPlot1 = plt.axes([.1,.1,.8,.75])

colors=['g','y','b']
crops=['Corn','Soy','Sorghum']
plt.plot([5,9],[multiCorr[0],multiCorr[0]],'-k',linewidth=3,label='Multivariate Regression')
plt.plot(xint,corrs[0,:,0],'^--k',label='NDVI')
plt.plot(xint,corrs[0,:,1],'s-.k',label='EVI')
plt.plot(xint,-1*corrs[0,:,2],'+:k',label='NDWI*(-1)')
for cp in range(3):
	plt.plot([5,9],[multiCorr[cp],multiCorr[cp]],'-'+colors[cp],linewidth=3,label=crops[cp])
for cp in range(3):
	plt.plot(xint,corrs[cp,:,0],'^--'+colors[cp])
	plt.plot(xint,corrs[cp,:,1],'s-.'+colors[cp])
	plt.plot(xint,-1*corrs[cp,:,2],'+:'+colors[cp])
	plt.legend()
plt.ylim(-.20,1.0001)
ax.set_xticks([5,6,7,8,9])
ax.set_xticklabels(['May','June','July','August','September'])
plt.title('Correlations Over Time')
plt.ylabel('Correlations to Corn Yield')
plt.grid(True)
plt.savefig(wdfigs+'Illinois/Illinois_corrs_over_time.pdf',dpi=700)
exit()


plt.clf()
ax=plt.gca()
plt.plot(xint,slopes[0,:,0],'^--k',label='NDVI')
plt.plot(xint,slopes[0,:,1],'s-.k',label='EVI')
plt.plot(xint,-1*slopes[0,:,2],'+:k',label='NDWI*(-1)')
plt.plot([5,9],[-10,-10],'-'+colors[0],label='Corn')
plt.plot([5,9],[-10,-10],'-'+colors[1],label='Soy')
plt.plot([5,9],[-10,-10],'-'+colors[2],label='Sorghum')
for cp in range(3):
	plt.plot(xint,slopes[cp,:,0],'^--'+colors[cp])
	plt.plot(xint,slopes[cp,:,1],'s-.'+colors[cp])
	plt.plot(xint,-1*slopes[cp,:,2],'+:'+colors[cp])
plt.legend(loc='lower center')
ax.set_xticks([5,6,7,8,9])
ax.set_xticklabels(['May','June','July','August','September'])
plt.ylim([-1.,5.2])
plt.title('Slopes Over Time')
plt.ylabel('Crop Yields/(Satellite Index *100), scaled by crop\'s std dev')
plt.grid(True)
plt.savefig(wdfigs+'Illinois/Illinois_slopes_over_time',dpi=700)

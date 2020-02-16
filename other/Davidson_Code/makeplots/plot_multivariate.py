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
from sklearn import linear_model
from operator import and_
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'
wdvars='/Users/lilllianpetersen/saved_vars/'
wdfigs='/Users/lilllianpetersen/figures/'

plt.clf()

xMulti=np.load(wdvars+'Illinois/xMulti.npy')
ydata=np.load(wdvars+'Illinois/ydataMulti.npy')

##############################
# Plot the 3D fig
##############################
def plot_figs(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(9, figsize=(6, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(X_train[:, 0], X_train[:, 1], ydata, c='b', marker='.')
    ax.plot_surface(np.array([[-15, -15], [15, 15]]),
                    np.array([[-15, 15], [-15, 15]]),
                    clf.predict(np.array([[-15, -15, 15, 15],
                                          [-15, 15, -15, 15]]).T
                                ).reshape((2, 2)),
                    color='g',
                    alpha=.5)
    ax.set_xlabel('August NDVI')
    ax.set_xlim([-15,15])
    ax.set_ylabel('August EVI')
    ax.set_ylim([-15,15])
    ax.set_zlabel('Illinois Corn Yield')
    #ax.w_xaxis.set_ticklabels([])
    #ax.w_yaxis.set_ticklabels([])
    #ax.w_zaxis.set_ticklabels([])
    plt.title('Multivariate Regression')
    plt.savefig(wdfigs+'Illinois/multivariet_regression_ndviAnom',dpi=700)


#Generate the three different figures from different views

Xplot=np.zeros(shape=(xMulti.shape[0],2))
Xplot[:,0]=xMulti[:,3]*100
Xplot[:,1]=xMulti[:,7]*100

ols=sklearn.linear_model.LinearRegression()
ols.fit(Xplot,ydata)

elev = 20
azim = -50
plot_figs(1, elev, azim, Xplot, ols)

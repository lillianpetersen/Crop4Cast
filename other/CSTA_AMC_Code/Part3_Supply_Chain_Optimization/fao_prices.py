
import csv
from math import sqrt
from sys import exit
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import os
from scipy.stats import norm
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

try:
    wddata='/Users/lilllianpetersen/iiasa/data/supply_chain/'
    wdfigs='/Users/lilllianpetersen/iiasa/figs/supply_chain/'
    wdvars='/Users/lilllianpetersen/iiasa/saved_vars/supply_chain/'
    f=open(wddata+'population/CAPITALVERSIONcasenumbers.csv','r')
except:
    wddata='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/data/'
    wdfigs='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/figs/'
    wdvars='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/vars/'

subsaharancountry = np.load(wdvars+'subsaharancountry.npy')

subsaharancountry[subsaharancountry=='Congo']='DRC'
subsaharancountry[subsaharancountry=='Congo (Republic of the)']='Congo'
subsaharancountry[subsaharancountry=="Cote d'Ivoire"]='Ivory Coast'

crops = np.array(['Barley','Beans, green','Beans, dry','Cassava','Groundnuts, with shell','Lentils','Maize','Millet','Oats','Rice, paddy','Sorghum','Soybeans','Sugar cane','Sweet potatoes','Taro (cocoyam)','Wheat','Yams'])

pricePerTon = np.zeros(shape = (len(subsaharancountry),len(crops),6))
pricePerTonMask = np.ones(shape = (len(subsaharancountry),len(crops),6),dtype=bool)
with open(wddata+'FAO_prices/FAOSTAT_data_4-2-2019.csv', 'rb') as csvfile:
    line=0
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for tmp in reader:
        line+=1
        country = tmp[3]
        if np.amax(country==subsaharancountry)==0:
            continue
        icountry = np.where(country==subsaharancountry)[0][0]

        crop = tmp[7]
        if np.amax(crop==crops)==0:
            continue
        icrop = np.where(crop==crops)[0][0]

        year = int(tmp[9])
        y = year-2012

        price = float(tmp[11])

        pricePerTon[icountry,icrop,y] = price
        pricePerTonMask[icountry,icrop,y] = 0

pricePerTon = np.ma.masked_array(pricePerTon,pricePerTonMask)

avgPricePerTon = np.zeros(shape=(len(subsaharancountry),len(crops)))
avgMask = np.ones(shape=(len(subsaharancountry),len(crops)),dtype=bool)
for icountry in range(len(subsaharancountry)):
    for icrop in range(len(crops)):
        if len(np.where(pricePerTonMask[icountry,icrop]==False)[0])<3:
            continue
        avgPricePerTon[icountry,icrop] = np.ma.mean(pricePerTon[icountry,icrop])
        avgMask[icountry,icrop] = 0

#avgPricePerTon = np.ma.masked_array(avgPricePerTon,avgMask)
crops = np.array(['Barley','Beans (green)','Beans (dry)','Cassava','Groundnuts (with shell)','Lentils','Maize','Millet','Oats','Rice (paddy)','Sorghum','Soybeans','Sugar cane','Sweet potatoes','Taro (cocoyam)','Wheat','Yams'])

for icountry in range(len(subsaharancountry)):
    country = subsaharancountry[icountry]
    if np.ma.is_masked(np.amax(avgPricePerTon[icountry])):
        continue
    f = open(wddata + 'FAO_prices/'+country+'_avgPrice_perTon.csv','w')
    for icrop in range(len(crops)):
        f.write(crops[icrop]+','+str(avgPricePerTon[icountry,icrop])+'\n')
    #np.savetxt(wddata + 'FAO_prices/'+country+'_avgPrice_perTon.csv', np.c_[crops,avgPricePerTon[icountry]])






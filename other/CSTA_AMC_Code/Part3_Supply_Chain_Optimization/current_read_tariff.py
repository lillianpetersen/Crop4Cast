################################################################
# Reads in World Bank tariff data
################################################################

from pulp import *
import math
import json
import numpy as np
import re
from sys import exit
import matplotlib.cm as cm
import matplotlib.pyplot as plt

try:
	wddata='/Users/lilllianpetersen/iiasa/data/supply_chain/'
	wdfigs='/Users/lilllianpetersen/iiasa/figs/'
	wdvars='/Users/lilllianpetersen/iiasa/saved_vars/supply_chain/'
	f=open(wddata+'trading_across_borders2017.csv','r')
except:
	wddata='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/data/'
	wdfigs='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/figs/'
	wdvars='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/vars/'

countrycosted=[]
capitalcosted=[]
f=open(wddata+'foodstuffs/current_prices.csv')
i=-1
for line in f:
    i+=1
    tmp=line.split(',')
    countrycosted.append(tmp[0])
    capitalcosted.append(tmp[4])
countrycosted=np.array(countrycosted)
countrycosted[countrycosted=='Congo']='DRC'
countrycosted[countrycosted=='Congo (Republic of the)']='Congo'
countrycosted[countrycosted=="I_Cote d'Ivoire"]='I_Ivory Coast'
countrycosted[countrycosted=="Cote d'Ivoire"]='Ivory Coast'


f = open(wddata+'tariff/world_bank_tariff/tariff.csv')
i=-2
countries=[]
tariff10=np.zeros(shape=(len(countrycosted),10)) # 2008-2018
tariff=np.zeros(shape=(len(countrycosted))) # 2008-2018
for line in f:
    i+=1
    if i==-1:
        continue
    line=line[:-2]
    line=line.replace('"','')
    tmp=np.array(line.split(','))
    country=tmp[0]
    if np.amax(country==np.array(countrycosted[:]))==0:
        continue
    countries.append(country)
    icountry=np.where(countrycosted==country)
    j=55 # 2007
    for y in range(10):
        j+=1
        try:
            tariff10[icountry,y]=float(tmp[j])
        except:
            continue
    tariff[icountry]=np.mean(tariff10[icountry][tariff10[icountry]!=0])

tariff[:24] = tariff[:24]-np.mean(tariff[:24])
tariff[:24] = tariff[:24]+7.0


np.save(wdvars+'current_tariff_by_country.npy',tariff)







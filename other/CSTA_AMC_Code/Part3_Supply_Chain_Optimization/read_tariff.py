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
    wdvars='/Users/lilllianpetersen/iiasa/saved_vars/'
    f=open(wddata+'trading_across_borders2017.csv','r')
except:
    wddata='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/data/'
    wdfigs='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/figs/'
    wdvars='C:/Users/garyk/Documents/code/riskAssessmentFromPovertyEstimations/supply_chain/vars/'

subsaharancountry = np.load(wdvars+'supply_chain/subsaharancountry.npy')
subsaharancountry[subsaharancountry=='Congo']='Congo (DRC)'
subsaharancountry[subsaharancountry=='Congo (Republic of the)']='Congo'

countrycosted=np.load(wdvars+'supply_chain/countrycosted.npy')
countrycosted[countrycosted=='Congo']='Congo (DRC)'
countrycosted[countrycosted=='Congo (Republic of the)']='Congo'


f = open(wddata+'tariff/world_bank_tariff/tariff.csv')
i=-2
countries=[]
tariff10=np.zeros(shape=(len(subsaharancountry),10)) # 2008-2018
tariff=np.zeros(shape=(len(subsaharancountry))) # 2008-2018
for line in f:
    i+=1
    if i==-1:
        continue
    line=line[:-2]
    line=line.replace('"','')
    tmp=np.array(line.split(','))
    country=tmp[0]
    if np.amax(country==np.array(subsaharancountry[:]))==0:
        continue
    countries.append(country)
    icountry=np.where(subsaharancountry==country)
    j=55 # 2007
    for y in range(10):
        j+=1
        try:
            tariff10[icountry,y]=float(tmp[j])
        except:
            continue
    tariff[icountry] = np.mean(tariff10[icountry][tariff10[icountry]!=0])
tariff[np.isnan(tariff)] = 0

#tariff[:24] = tariff[:24]-np.mean(tariff[:24])
#tariff[:24] = tariff[:24]+7.0


#np.save(wdvars+'supply_chain/tariff_by_country.npy',tariff)

africanCountries = np.load(wdvars+'country_correlates/africanCountries.npy')

tariff1 = np.zeros(shape=(43))
for i in range(len(subsaharancountry)):
    country = subsaharancountry[i]
    if country[:2]=='I_': continue
    if country=='DRC': country='Congo (DRC)'
    if country=='Ivory Coast': country="Cote d'Ivoire"
    
    p = np.where(country==africanCountries)[0][0]
    tariff1[p] = tariff[i]

tariff1Full = np.zeros(shape=(43,22))
for y in range(22):
	tariff1Full[:,y] = tariff1
np.save(wdvars+'country_correlates/tariff.npy',tariff1Full)







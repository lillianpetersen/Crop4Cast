import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import math
import sys
from sys import exit

wd='/Users/lilllianpetersen/Google Drive/science_fair/'
wddata='/Users/lilllianpetersen/data/'
wdvars='/Users/lilllianpetersen/saved_vars/'
wdfigs='/Users/lilllianpetersen/figures/'

countylats=np.load(wdvars+'county_lats.npy')
countylons=np.load(wdvars+'county_lons.npy')
countyName=np.load(wdvars+'countyName.npy')
stateName=np.load(wdvars+'stateName.npy')

nyears=17
nName=['15n','16n']
makePlots=False


for icounty in range(len(countylats)):

	clat=countylats[icounty]
	clon=countylons[icounty]
	cName=countyName[icounty].title()
	cName=cName.replace(' ','_')
	sName=stateName[icounty].title()

	if sName!='Illinois':
		continue
	if cName!='Mercer':
		continue
	#if cName!='Alexander' and cName!='Cass' and cName!='Mason' and cName!='Menard' and cName!='Morgan' and cName!='Sangamon':
	#	continue

	for n in range(2):
		if n==1: 
			continue

		try:
				ndviClimo=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviClimoUnprocessed.npy')
				ndviMonthAvgU=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviMonthAvgUnprocessed.npy')
				
				eviClimo=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviClimoUnprocessed.npy')
				eviMonthAvgU=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviMonthAvgUnprocessed.npy')

				ndwiClimo=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiClimoUnprocessed.npy')
				ndwiMonthAvgU=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiMonthAvgUnprocessed.npy')

		except:
			print 'no',nName[n],'for', cName
			continue



		vlen=ndviMonthAvgU.shape[2]
		hlen=ndviMonthAvgU.shape[3]

		if ndviMonthAvgU.shape[1]==12:
			print cName,'has 12 months for',nName[n]
			exit()

		if np.amax(eviClimo)!=0 and np.amax(ndwiClimo)!=0:
			print 'good'
			continue
		print 'running',nName[n],'for', cName

		for m in range(5):
			for v in range(vlen):
				for h in range(hlen):
					eviClimo[m,v,h]=np.sum(eviMonthAvgU[:,m,v,h])
					ndwiClimo[m,v,h]=np.sum(ndwiMonthAvgU[:,m,v,h])

		#if cName=='Alexander' or cName=='Cass' or cName=='Mason' or cName=='Menard' or cName=='Morgan' or cName=='Sangamon':
		#	for m in range(5):
		#		for v in range(vlen):
		#			for h in range(vlen):
		#				ndviClimo[:,v,h]=np.sum(ndviMonthAvgU[:,:,v,h])


		###########################################################
		# save variables
		###########################################################
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviClimoUnprocessed',ndviClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviClimoUnprocessed',eviClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiClimoUnprocessed',ndwiClimo)

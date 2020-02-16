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
	if cName!='Bureau':
		continue
	#if cName!='Cass' and cName!='Mason' and cName!='Menard' and cName!='Morgan' and cName!='Sangamon':
	#	continue

	for n in range(2):
		if n==0: 
			#sys.modules[__name__].__dict__.clear()
			continue

		try:
			if n==1: #or cName=='Adams' or cName=='Cass':
				ndviClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/all_months/ndviClimoUnprocessed.npy')
				climoCounterAllAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/all_months/climoCounterUnprocessed.npy')
				ndviMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/all_months/ndviMonthAvgUnprocessed.npy')
				
				eviClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/all_months/eviClimoUnprocessed.npy')
				eviMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/all_months/eviMonthAvgUnprocessed.npy')

				ndwiClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/all_months/ndwiClimoUnprocessed.npy')
				ndwiMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/all_months/ndwiMonthAvgUnprocessed.npy')

			elif n==0:
				ndviClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviClimoUnprocessed.npy')
				climoCounterAllAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/climoCounterUnprocessed.npy')
				ndviMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviMonthAvgUnprocessed.npy')
				
				eviClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviClimoUnprocessed.npy')
				eviMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviMonthAvgUnprocessed.npy')

				ndwiClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiClimoUnprocessed.npy')
				ndwiMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiMonthAvgUnprocessed.npy')
				countyMaskNotBoolAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/countyMask.npy')

		except:
			print 'no',nName[n],'for', cName
			continue


		print 'running',nName[n],'for', cName

		vlen=ndviMonthAvgUAllMonths.shape[2]
		hlen=ndviMonthAvgUAllMonths.shape[3]

		if ndviMonthAvgUAllMonths.shape[1]==5:
			print cName,'has 5 month for',nName[n]
			continue		
		###########################################################
		# turn into 5 month long array
		###########################################################
		climoCounterAll=np.zeros(shape=(nyears,5,vlen,hlen))
		ndviMonthAvgU=np.zeros(shape=(nyears,5,vlen,hlen))
		eviMonthAvgU=np.zeros(shape=(nyears,5,vlen,hlen))
		ndwiMonthAvgU=np.zeros(shape=(nyears,5,vlen,hlen))

		ndviClimo=np.zeros(shape=(5,vlen,hlen))
		eviClimo=np.zeros(shape=(5,vlen,hlen))
		ndwiClimo=np.zeros(shape=(5,vlen,hlen))

		for y in range(nyears):
			for m in range(5):
				climoCounterAll[y,m,:,:]=climoCounterAllAllMonths[y,m+4,:,:]
				ndviMonthAvgU[y,m,:,:]=ndviMonthAvgUAllMonths[y,m+4,:,:]
				eviMonthAvgU[y,m,:,:]=eviMonthAvgUAllMonths[y,m+4,:,:]
				ndwiMonthAvgU[y,m,:,:]=ndwiMonthAvgUAllMonths[y,m+4,:,:]

		for m in range(5):
			ndviClimo[m,:,:]=ndviClimoAllMonths[m+4,:,:]
			eviClimo[m,:,:]=eviClimoAllMonths[m+4,:,:]
			ndwiClimo[m,:,:]=ndwiClimoAllMonths[m+4,:,:]


		###########################################################
		# save 5 month variables
		###########################################################
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviClimoUnprocessed',ndviClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviClimoUnprocessed',eviClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiClimoUnprocessed',ndwiClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/climoCounterUnprocessed',climoCounterAll)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviMonthAvgUnprocessed',ndviMonthAvgU)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviMonthAvgUnprocessed',eviMonthAvgU)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiMonthAvgUnprocessed',ndwiMonthAvgU)


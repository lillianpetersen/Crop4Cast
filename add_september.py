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

	if cName!='De_Witt':
		continue

	for n in range(2):

		try:
			ndviClimoSep=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/september/ndviClimoUnprocessed.npy')
			climoCounterAllSep=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/september/climoCounterUnprocessed.npy')
			ndviMonthAvgUSep=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/september/ndviMonthAvgUnprocessed.npy')
			
			eviClimoSep=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/september/eviClimoUnprocessed.npy')
			eviMonthAvgUSep=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/september/eviMonthAvgUnprocessed.npy')

			ndwiClimoSep=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/september/ndwiClimoUnprocessed.npy')
			ndwiMonthAvgUSep=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/september/ndwiMonthAvgUnprocessed.npy')


			ndviClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/no_september/ndviClimoUnprocessed.npy')
			climoCounterAllAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/no_september/climoCounterUnprocessed.npy')
			ndviMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/no_september/ndviMonthAvgUnprocessed.npy')
			
			eviClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/no_september/eviClimoUnprocessed.npy')
			eviMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/no_september/eviMonthAvgUnprocessed.npy')

			ndwiClimoAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/no_september/ndwiClimoUnprocessed.npy')
			ndwiMonthAvgUAllMonths=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/no_september/ndwiMonthAvgUnprocessed.npy')

		except:
			print 'no',nName[n],'for', cName
			continue


		print 'running',nName[n],'for', cName

		vlen=climoCounterAllAllMonths.shape[2]
		hlen=climoCounterAllAllMonths.shape[3]
		
		###########################################################
		# add september
		###########################################################
		for y in range(nyears):
			climoCounterAllAllMonths[y,8,:,:]=climoCounterAllSep[y,8,:,:]
			ndviMonthAvgUAllMonths[y,8,:,:]=ndviMonthAvgUSep[y,8,:,:]
			eviMonthAvgUAllMonths[y,8,:,:]=eviMonthAvgUSep[y,8,:,:]
			ndwiMonthAvgUAllMonths[y,8,:,:]=ndwiMonthAvgUSep[y,8,:,:]

		ndviClimoAllMonths[8]=ndviClimoSep[8]
		eviClimoAllMonths[8]=eviClimoSep[8]
		ndwiClimoAllMonths[8]=ndwiClimoSep[8]

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


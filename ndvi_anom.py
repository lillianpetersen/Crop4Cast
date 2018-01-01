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
wdvars='/Users/lilllianpetersen/saved_vars/'
wdfigs='/Users/lilllianpetersen/figures/'

countylats=np.load(wdvars+'county_lats.npy')
countylons=np.load(wdvars+'county_lons.npy')
countyName=np.load(wdvars+'countyName.npy')
stateName=np.load(wdvars+'stateName.npy')

nyears=16
nName=['15n','16n']
makePlots=False

ndviAnom=np.zeros(shape=(3143,nyears,12))
eviAnom=np.zeros(shape=(3143,nyears,12))
ndwiAnom=np.zeros(shape=(3143,nyears,12))

for icounty in range(len(countylats)):

	clat=countylats[icounty]
	clon=countylons[icounty]
	cName=countyName[icounty].title()
	sName=stateName[icounty].title()

	if sName!='Illinois':
		continue
	if clat<38:
		continue

	print '\n',cName

	goodn=np.ones(shape=(2),dtype=bool)

	counterSum=np.zeros(shape=(nyears,12))
	ndviAnomSum=np.zeros(shape=(nyears,12))
	eviAnomSum=np.zeros(shape=(nyears,12))
	ndwiAnomSum=np.zeros(shape=(nyears,12))

	countyMask=np.ones(shape=(vlen,hlen))

	for n in range(2):

		try:
			ndviClimo=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviClimoUnprocessed.npy')
			climoCounterAll=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/climoCounterUnprocessed.npy')
			ndviMonthAvgU=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviMonthAvgUnprocessed.npy')
			
			eviClimo=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviClimoUnprocessed.npy')
			eviMonthAvgU=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviMonthAvgUnprocessed.npy')

			ndwiClimo=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiClimoUnprocessed.npy')
			ndwiMonthAvgU=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiMonthAvgUnprocessed.npy')
			countyMaskNotBool=np.load(wdvars+sName+'/'+cName+'/'+nName[n]+'/countyMask.npy')
		except:
			print 'no',nName[n],'for', cName
			goodn[n]=False
			continue

		if climoCounterAll.shape[0]==1:
			print 'ONLY ONE YEAR!!!!!', nName[n]
			continue

		print 'running',nName[n]
		
		vlen=climoCounterAll.shape[2]
		hlen=climoCounterAll.shape[3]

		countyMask=np.zeros(shape=(vlen,hlen),dtype=bool)
		ndviMonthAvg=np.zeros(shape=(ndviMonthAvgU.shape))
		eviMonthAvg=np.zeros(shape=(eviMonthAvgU.shape))
		ndwiMonthAvg=np.zeros(shape=(ndwiMonthAvgU.shape))

		for v in range(vlen):
			for h in range(hlen):
				countyMask[v,h]=bool(countyMaskNotBool[v,h])
		
		climoCounter=np.zeros(shape=(12,vlen,hlen)) # number of days in every of each month
		ndviAnomAllPix=np.zeros(shape=(nyears,12,vlen,hlen))
		eviAnomAllPix=np.zeros(shape=(nyears,12,vlen,hlen))
		ndwiAnomAllPix=np.zeros(shape=(nyears,12,vlen,hlen))
		
		for m in range(4,8):
			if m!=4 and m!=5 and m!=6 and m!=7:
				continue
			for v in range(vlen):
				for h in range(hlen):
					if countyMask[v,h]==1:
						continue
					climoCounter[m,v,h]=np.sum(climoCounterAll[:,m,v,h])
		
					ndviClimo[m,v,h]=ndviClimo[m,v,h]/climoCounter[m,v,h]
			 		eviClimo[m,v,h]=eviClimo[m,v,h]/climoCounter[m,v,h]
					ndwiClimo[m,v,h]=ndwiClimo[m,v,h]/climoCounter[m,v,h]
		
				 
		for y in range(nyears):
			for m in range(4,8):
				for v in range(vlen):
					for h in range(hlen):
						if countyMask[v,h]==1:
							continue
						ndviMonthAvg[y,m,v,h]=ndviMonthAvgU[y,m,v,h]/climoCounterAll[y,m,v,h]
						eviMonthAvg[y,m,v,h]=eviMonthAvgU[y,m,v,h]/climoCounterAll[y,m,v,h]
						ndwiMonthAvg[y,m,v,h]=ndwiMonthAvgU[y,m,v,h]/climoCounterAll[y,m,v,h]
		
						ndviAnomAllPix[y,m,v,h]=ndviMonthAvg[y,m,v,h]-ndviClimo[m,v,h]
						eviAnomAllPix[y,m,v,h]=eviMonthAvg[y,m,v,h]-eviClimo[m,v,h]
						ndwiAnomAllPix[y,m,v,h]=ndwiMonthAvg[y,m,v,h]-ndwiClimo[m,v,h]

	#	if makePlots:
	#		if not os.path.exists(wdfigs+sName+'/'+cName):
	#			os.makedirs(wdfigs+sName+'/'+cName)

	#		plt.clf()
	#		plt.imshow(ndviClimo[7,:,:], vmin=-.6, vmax=.9)
	#		plt.colorbar()
	#		plt.title('ndvi August Climatology Ohio')
	#		plt.savefig(wdfigs+sName+'/'+cName+'/ndviClimo_Aug',dpi=700)
	#		
	#		plt.clf()
	#		plt.imshow(eviClimo[7,:,:], vmin=-.6, vmax=.9)
	#		plt.colorbar()
	#		plt.title('evi August Climatology Ohio')
	#		plt.savefig(wdfigs+sName+'/'+cName+'/eviClimo_Aug',dpi=700)
	#		
	#		plt.clf()
	#		plt.imshow(ndwiClimo[7,:,:], vmin=-.6, vmax=.9)
	#		plt.colorbar()
	#		plt.title('ndwi August Climatology Ohio')
	#		plt.savefig(wdfigs+sName+'/'+cName+'/ndwiClimo_Aug',dpi=700)
	#	
	#	if makePlots:
	#		plt.clf()
	#		plt.figure(1,figsize=(3,3))
	#		plt.plot(np.ma.compressed(ndviAnomAllPix[:,4:8,20,11]),'*-b')
	#		plt.plot(np.ma.compressed(ndviAnomAllPix[:,:,20,11]),'*-b')
	#		plt.ylim(-.25,.25)
	#		plt.title('ndvi Anomaly for pixel 20, 11')
	#		plt.savefig(wdfigs+sName+'/'+cName+'/ndviAnomAllPix_20_11',dpi=700)
	#		
	#		plt.clf()
	#		plt.figure(1,figsize=(3,3))
	#		plt.plot(np.ma.compressed(ndviAnomAllPix[:,4:8,50,30]),'*-b')
	#		plt.plot(np.ma.compressed(ndviAnomAllPix[:,:,50,30]),'*-b')
	#		plt.ylim(-.25,.25)
	#		plt.title('ndvi Anomaly for pixel 50, 30')
	#		plt.savefig(wdfigs+sName+'/'+cName+'/ndviAnomAllPix_50_30',dpi=700)
	#
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviClimo',ndviClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/climoCounter',climoCounter)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviMonthAvg',ndviMonthAvg)
		
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviClimo',eviClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviMonthAvg',eviMonthAvg)
		
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiClimo',ndwiClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiMonthAvg',ndwiMonthAvg)
	
		for y in range(nyears):
			for m in range(4,8):
				for v in range(vlen):
					for h in range(hlen):
						if countyMask[v,h]==1:
							continue
						if math.isnan(ndviAnomAllPix[y,m,v,h])==False:
							counterSum[y,m]+=climoCounterAll[y,m,v,h]
							ndviAnomSum[y,m]+=ndviAnomAllPix[y,m,v,h]
							eviAnomSum[y,m]+=eviAnomAllPix[y,m,v,h]
							ndwiAnomSum[y,m]+=ndwiAnomAllPix[y,m,v,h]
	
	if np.sum(goodn)==0:
		continue

	for y in range(nyears):
		for m in range(4,8):
			ndviAnom[icounty,y,m]=ndviAnomSum[y,m]/counterSum[y,m]
			eviAnom[icounty,y,m]=eviAnomSum[y,m]/counterSum[y,m]
			ndwiAnom[icounty,y,m]=ndwiAnomSum[y,m]/counterSum[y,m]

	print ndviAnom[icounty,3,4:8]

	#if makePlots:
	#	plt.clf()
	#	plt.figure(1,figsize=(3,3))
	#	plt.plot(np.ma.compressed(ndviAnom[:,4:8]),'*-b')
	#	plt.title('ndvi Anomaly Ohio')
	#	plt.ylim([-.08,.08])
	#	plt.savefig(wdfigs+sName+'/'+cName+'/ndviAnom',dpi=700)
	#	
	#	plt.clf()
	#	plt.figure(1,figsize=(3,3))
	#	plt.plot(np.ma.compressed(eviAnom[:,4:8]),'*-b')
	#	plt.title('evi Anomaly Ohio')
	#	plt.ylim([-.08,.08])
	#	plt.savefig(wdfigs+sName+'/'+cName+'/eviAnom',dpi=700)
	#	
	#	plt.clf()
	#	plt.figure(1,figsize=(3,3))
	#	plt.plot(np.ma.compressed(ndwiAnom[:,4:8]),'*-b')
	#	plt.title('ndwi Anomaly Ohio')
	#	plt.ylim([-.08,.08])
	#	plt.savefig(wdfigs+sName+'/'+cName+'/ndwiAnom',dpi=700)
	#exit()
	


np.save(wdvars+sName+'/ndviAnom',ndviAnom)
np.save(wdvars+sName+'/eviAnom',eviAnom)
np.save(wdvars+sName+'/ndwiAnom',ndwiAnom)

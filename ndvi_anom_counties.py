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

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

colors = [(.4,0,.6), (0,0,.7), (0,.6,1), (.9,.9,1), (1,.8,.8), (1,1,0), (.8,1,.5), (.1,.7,.1), (.1,.3,.1)]
my_cmap = make_cmap(colors)
#my_cmap_r=make_cmap(colors[::-1])

colors = [(128, 66, 0), (255, 230, 204), (255,255,255), (204, 255, 204), (0,100,0)]
my_cmap_gwb = make_cmap(colors,bit=True)
#my_cmap_gwb_r=make_cmap(colors[::-1],bit=True)


counties=False
tiles=True


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

ndviAnom=np.zeros(shape=(3143,nyears,12))
eviAnom=np.zeros(shape=(3143,nyears,12))
ndwiAnom=np.zeros(shape=(3143,nyears,12))

ndviAvg=np.zeros(shape=(3143,nyears,12))
eviAvg=np.zeros(shape=(3143,nyears,12))
ndwiAvg=np.zeros(shape=(3143,nyears,12))

for icounty in range(len(countylats)):

	clat=countylats[icounty]
	clon=countylons[icounty]
	cName=countyName[icounty].title()
	sName=stateName[icounty].title()

	if sName!='Illinois':
		continue

	print '\n',sName,cName

	goodn=np.ones(shape=(2),dtype=bool)

	counterSum=np.zeros(shape=(nyears,12))
	counterSumforAvg=np.zeros(shape=(nyears,12))
	ndviAnomSum=np.zeros(shape=(nyears,12))
	eviAnomSum=np.zeros(shape=(nyears,12))
	ndwiAnomSum=np.zeros(shape=(nyears,12))

	ndviAvgSum=np.zeros(shape=(nyears,12))
	eviAvgSum=np.zeros(shape=(nyears,12))
	ndwiAvgSum=np.zeros(shape=(nyears,12))


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
			print 'no',sName,'for', cName
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
		
		for m in range(4,9):
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

		monthName=['January','Febuary','March','April','May','June','July','August','September','October','November','December']
		#for m in range(4,9):
		#	plt.clf()
		#	plt.imshow(ndviClimo[m,:,:],cmap=my_cmap,vmin=-.3,vmax=.8)
		#	plt.colorbar()
		#	plt.title(sName+' '+monthName[m]+' NDVI Climatology')
		#	plt.savefig(wdfigs+sName+'/'+str(m)+monthName[m]+'_ndvi_climatology')

				 
		for y in range(nyears):
			for m in range(4,9):
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


		#for m in range(4,9):
		#	plt.clf()
		#	plt.imshow(ndviMonthAvg[2,m,:,:],cmap=my_cmap,vmin=-.3,vmax=.8)
		#	plt.colorbar()
		#	plt.title(sName+' '+monthName[m]+' 2015 Average NDVI')
		#	plt.savefig(wdfigs+sName+'/'+str(m)+monthName[m]+'2015_ndvi_month_avg',dpi=700)

		#plt.clf()
		#plt.imshow(ndviMonthAvg[14,7,:,:],cmap=my_cmap,vmin=0.,vmax=.85)
		#plt.colorbar()
		##plt.title(cName+' County 2014 August Monthly Average')
		#plt.title(sName+' 2014 August Monthly Average')
		#plt.savefig(wdfigs+sName+'/7_ndviMonthAvg_2014',dpi=700)

		#plt.clf()
		#plt.imshow(ndviMonthAvg[12,7,:,:],cmap=my_cmap,vmi1=0.,vmax=.85)
		#plt.colorbar()
		##plt.title(cName+' County 2012 August Monthly Average')
		#plt.title(sName+' 2012 August Monthly Average')
		#plt.savefig(wdfigs+sName+'/7_ndviMonthAvg_2012',dpi=700)

		#plt.clf()
		#plt.imshow(ndviClimo[7,:,:],cmap=my_cmap,vmin=0.,vmax=.85)
		#plt.colorbar()
		##plt.title(cName+' County August Climatology')
		#plt.title(sName+' August Climatology')
		#plt.savefig(wdfigs+sName+'/7_ndviClimo_2014',dpi=700)
		#exit()

#		if makePlots:
#			if not os.path.exists(wdfigs+sName+'/'+cName):
#				os.makedirs(wdfigs+sName+'/'+cName)

#			plt.clf()
#			plt.imshow(ndviClimo[7,:,:], vmin=-.6, vmax=.9)
#			plt.colorbar()
#			plt.title('ndvi August Climatology Ohio')
#			plt.savefig(wdfigs+sName+'/'+cName+'/ndviClimo_Aug',dpi=700)
#			
#			plt.clf()
#			plt.imshow(eviClimo[7,:,:], vmin=-.6, vmax=.9)
#			plt.colorbar()
#			plt.title('evi August Climatology Ohio')
#			plt.savefig(wdfigs+sName+'/'+cName+'/eviClimo_Aug',dpi=700)
#			
#			plt.clf()
#			plt.imshow(ndwiClimo[7,:,:], vmin=-.6, vmax=.9)
#			plt.colorbar()
#			plt.title('ndwi August Climatology Ohio')
#			plt.savefig(wdfigs+sName+'/'+cName+'/ndwiClimo_Aug',dpi=700)
#		
#		if makePlots:
#			plt.clf()
#			plt.figure(1,figsize=(3,3))
#			plt.plot(np.ma.compressed(ndviAnomAllPix[:,4:8,20,11]),'*-b')
#			plt.plot(np.ma.compressed(ndviAnomAllPix[:,:,20,11]),'*-b')
#			plt.ylim(-.25,.25)
#			plt.title('ndvi Anomaly for pixel 20, 11')
#			plt.savefig(wdfigs+sName+'/'+cName+'/ndviAnomAllPix_20_11',dpi=700)
#			
#			plt.clf()
#			plt.figure(1,figsize=(3,3))
#			plt.plot(np.ma.compressed(ndviAnomAllPix[:,4:8,50,30]),'*-b')
#			plt.plot(np.ma.compressed(ndviAnomAllPix[:,:,50,30]),'*-b')
#			plt.ylim(-.25,.25)
#			plt.title('ndvi Anomaly for pixel 50, 30')
#			plt.savefig(wdfigs+sName+'/'+cName+'/ndviAnomAllPix_50_30',dpi=700)
#
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviClimo',ndviClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/climoCounter',climoCounter)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndviMonthAvg',ndviMonthAvg)
		
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviClimo',eviClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/eviMonthAvg',eviMonthAvg)
		
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiClimo',ndwiClimo)
		np.save(wdvars+sName+'/'+cName+'/'+nName[n]+'/ndwiMonthAvg',ndwiMonthAvg)

		for y in range(nyears):
			for m in range(4,9):
				for v in range(vlen):
					for h in range(hlen):
						if countyMask[v,h]==1:
							continue
						if math.isnan(ndviAnomAllPix[y,m,v,h])==False and ndviAnomAllPix[y,m,v,h]!=0.:
							counterSum[y,m]+=1
							ndviAnomSum[y,m]+=ndviAnomAllPix[y,m,v,h]
							eviAnomSum[y,m]+=eviAnomAllPix[y,m,v,h]
							ndwiAnomSum[y,m]+=ndwiAnomAllPix[y,m,v,h]

						if math.isnan(ndviMonthAvg[y,m,v,h])==False and ndviMonthAvg[y,m,v,h]!=0.:
							counterSumforAvg[y,m]+=1
							ndviAvgSum[y,m]+=ndviMonthAvg[y,m,v,h]
							eviAvgSum[y,m]+=eviMonthAvg[y,m,v,h]
							ndwiAvgSum[y,m]+=ndwiMonthAvg[y,m,v,h]


	for y in range(nyears):
		for m in range(4,9):
			ndviAnom[icounty,y,m]=ndviAnomSum[y,m]/counterSum[y,m]
			eviAnom[icounty,y,m]=eviAnomSum[y,m]/counterSum[y,m]
			ndwiAnom[icounty,y,m]=ndwiAnomSum[y,m]/counterSum[y,m]
	
			ndviAvg[icounty,y,m]=ndviAvgSum[y,m]/counterSumforAvg[y,m]
			eviAvg[icounty,y,m]=eviAvgSum[y,m]/counterSumforAvg[y,m]
			ndwiAvg[icounty,y,m]=ndwiAvgSum[y,m]/counterSumforAvg[y,m]
	
	print ndviAnom[icounty,3,:]
	print ndviAvg[icounty,3,:]

	np.save(wdvars+sName+'/ndviAnom',ndviAnom)
	np.save(wdvars+sName+'/eviAnom',eviAnom)
	np.save(wdvars+sName+'/ndwiAnom',ndwiAnom)
	
	np.save(wdvars+sName+'/ndviAvg',ndviAvg)
	np.save(wdvars+sName+'/eviAvg',eviAvg)
	np.save(wdvars+sName+'/ndwiAvg',ndwiAvg)

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

#ndviAvgPlot=np.ma.compressed(ndviAvg[:4])
#x=np.zeros(shape=(nyears,12))
#for y in range(nyears):
#	for m in range(4,9):
#		x[y,m]=(y+2013)+(m+1.5)/12
#x=np.ma.compressed(x)
#plt.clf()
#plt.plot(x,ndviAvgPlot,'b*-')
#plt.title('NDVI Monthly Average '+sName)
#plt.grid(True)
#plt.savefig(wdfigs+sName+'/ndvi_monthlyavg_over_time_'+sName,dpi=700)
#
#plt.clf()
#x=np.zeros(shape=(nyears,12))
#for y in range(nyears):
#	for m in range(4,9):
#		x[y,m]=m+1
#	plt.plot(x[y],ndviAvg[0,y],'b*-')
#
#plt.title('NDVI Monthly Average '+sName)
#plt.grid(True)
#plt.savefig(wdfigs+sName+'/ndvi_monthlyavg_months_'+sName,dpi=700)
#
#	
#ndviAnomPlot=np.ma.compressed(ndviAnom[:4])
#x=np.zeros(shape=(nyears,12))
#for y in range(nyears):
#	for m in range(4,9):
#		x[y,m]=(y+2013)+(m+1.5)/12
#x=np.ma.compressed(x)
#plt.clf()
#plt.plot(x,ndviAnomPlot,'b*-')
#plt.title('NDVI Monthly Anomaly '+sName)
#plt.grid(True)
#plt.savefig(wdfigs+sName+'/ndvi_monthlyAnom_over_time_'+sName,dpi=700)
#
#plt.clf()
#x=np.zeros(shape=(nyears,12))
#for y in range(nyears):
#	for m in range(4,9):
#		x[y,m]=m+1
#	plt.plot(x[y],ndviAnom[0,y],'b*-')
#
#plt.title('NDVI Monthly Anomaly '+sName)
#plt.grid(True)
#plt.savefig(wdfigs+sName+'/ndvi_monthlyAnom_months_'+sName,dpi=700)
#
#for y in range(5):
#	plt.clf()
#	plt.figure(1)
#	plt.imshow(ndviAnomAllPix[y,8,:,:]*100,cmap=my_cmap_gwb,vmin=-10,vmax=10)
#	plt.colorbar()
#	plt.title(sName+' '+monthName[8]+' '+str(y+2013)+' NDVI Anomaly *100')
#	plt.savefig(wdfigs+sName+'/'+str(8)+monthName[8]+str(y+2013)+'_ndvi_month_anom',dpi=700)


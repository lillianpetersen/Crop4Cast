import os
import matplotlib.pyplot as plt
import descarteslabs as dl
import numpy as np
import pickle
import sys
from sys import exit
import pandas as pd

####################
# Functions        #
####################
### Running mean/Moving average
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
    
def Variance(x):   
    '''function to compute the variance (std dev squared)'''
    xAvg=np.average(x)
    xOut=0.
    for k in range(len(x)):
        xOut=xOut+(x[k]-xAvg)**2
    xOut=xOut/(k+1)
    return xOut
####################        

wd='/Users/lilllianpetersen/Documents/science_fair/'

lonAll=[-57.864114, -57.345734, -56.0979, -95.156364, -123.6585, -110.580639, 111.233326]
latAll= [-13.458213, -12.748814,  -15.6014, 41.185114, 39.3592, 35.772751, 51.158285]
lon=lonAll[6]
lat=latAll[6]
vlen=1008
hlen=1008
start='2014-01-01'
end='2016-12-31'
country='US'
makePlots=0
#compute_ndvi(lon,lat,vlen,hlen,start,end,country,makePlots)     

clas=["" for x in range(256)]
f=open(wd+'data/ground_data.txt')                                
for line in f:
    tmp=line.split(',')
    classNum=int(tmp[0])
    clas[classNum]=tmp[1]
    
    
"""function to compute ndvi and make summary plots
variables: lon, lat, vlen, hlen, start, end, country, makePlots
"""

makePlots=bool(makePlots)
latsave=str(lat)
latsave=latsave.replace('.','-')
lonsave=str(lat)
lonsave=lonsave.replace('.','-')

matches=dl.places.find('united-states_california')
aoi = matches[0]
shape = dl.places.shape(aoi['slug'], geom='low')

padding = 8
valid_pix = vlen+2*padding
res = 120.0
#dltile = dl.raster.dltile_from_latlon(lat, lon, res, valid_pix, padding)
dltiles = dl.raster.dltiles_from_shape(res, valid_pix, padding, shape)

for tile in range(len(dltiles['features'])):
    dltile=dltiles['features'][tile]
    
    images = dl.metadata.search(
        const_id=["MO", "MY"],
        start_time=start,
        end_time=end,
        geom=dltile['geometry'],
        cloud_fraction=0.9,
        limit = 2000
        )
    
    n_images = len(images['features'])
    print('Number of image matches: %d' % n_images)
    k=n_images
    mo = dl.raster.get_bands_by_constellation("MO").keys()
    my = dl.raster.get_bands_by_constellation("MY").keys()
    avail_bands = set(mo).intersection(my)
    print('Available bands: %s' % ', '.join([a for a in avail_bands]))
    
    band_info = dl.raster.get_bands_by_constellation("MO")
    
    ####################
    # Define Variables #
    ####################
    ndviAll=-9999*np.ones(shape=(vlen,hlen,k+1)) 
    cloudAll=-9999*np.ones(shape=(vlen,hlen,k+1))  
    dayOfYear=np.zeros(shape=(k+1))
    year=np.zeros(shape=(k+1))
    month=np.zeros(shape=(k+1))
    day=np.zeros(shape=(k+1))
    plotYear=np.zeros(shape=(k+1))
    ndviHist=np.zeros(shape=(40,k+1))
    ndviAvg=np.zeros(shape=(k+1))
    xtime=[]
    ####################
    
    j=-1
    k=-1
    for feature in images['features']:
        j+=1
        # get the scene id
        scene = feature['id']
        
        ###############################################
        # NDVI
        ###############################################
        # load the image data into a numpy array
        try:
            valid_range = band_info['ndvi']['valid_range']
            physical_range = band_info['ndvi']['physical_range']
            arrNDVI, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['ndvi', 'alpha'],
                scales=[[valid_range[0], valid_range[1], physical_range[0], physical_range[1]], None],
                data_type='Float32'
                )
        except:
            print('ndvi: %s could not be retreived' % scene)
            continue
      
        ###############################################
        # Test for bad days
        ############################################### 
    
        #take out days without data       
        maskforNDVI = arrNDVI[:, :, 1] == 0
        if np.sum(maskforNDVI)==vlen*hlen:
            print 'continued'
            continue
        
        #######################
        # Get cloud data      #
        #######################
        try:
            valid_range = band_info['visual_cloud_mask']['valid_range']
            physical_range = band_info['visual_cloud_mask']['physical_range']
            arrCloud, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['visual_cloud_mask', 'alpha'],
                scales=[[valid_range[0], valid_range[1]], None],
                data_type='Float32'
                )
        except:
            print('cloud: %s could not be retreived' % scene)
            continue 
        #######################
        
        #### Only for Desert ####
        for v in range(vlen):
            for h in range(hlen):
                arrCloud[:,:,0]=0
        #### Only for Desert ####
        
        # take out days with too many clouds
        maskforCloud = arrCloud[:, :, 0] == 0
        if np.sum(maskforCloud)<0.1*(vlen*hlen):
            print 'clouds: continued'
            continue        
        k+=1
        
        ###############################################
        # time
        ############################################### 
        
        xtime.append(str(images['features'][j]['id'][5:15]))
        date=xtime[k]
        year[k]=xtime[k][0:4]
        month[k]=xtime[k][5:7]
        day[k]=xtime[k][8:10]
        dayOfYear[k]=(float(month[k])-1)*30+float(day[k])
        plotYear[k]=year[k]+dayOfYear[k]/365.0
        
        ###############################################
        # Back to NDVI
        ############################################### 
    
        print date, k
        sys.stdout.flush()
        maskforCloud = arrCloud[:, :, 0] != 0 
        #maskforCloud = arrCloud[:, :, 1] == 0 #for desert
        if makePlots:
            
            if not os.path.exists(r''+wd+'/figures/'+country+'/'+str(lon)+'_'+str(lat)):
                os.makedirs(r''+wd+'figures/'+country+'/'+str(lon)+'_'+str(lat))
            
            masked_ndvi = np.ma.masked_array(arrNDVI[:, :, 0], maskforCloud)
            plt.figure(figsize=[16,16])
            plt.imshow(masked_ndvi, cmap='jet', vmin=-1, vmax=1)
            plt.title('NDVI: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
            cb = plt.colorbar()
            cb.set_label("NDVI")
            plt.savefig('figures/'+country+'/'+str(lon)+'_'+str(lat)+'/ndvi_'+str(date))
            plt.clf()    
            
        for v in range(vlen):
            for h in range(hlen):
                ndviAll[v,h,k]=arrNDVI[v,h,0]
        
        
        ###############################################
        # Cloud
        ###############################################
            
        if makePlots:
            masked_cloud = np.ma.masked_array(arrCloud[:, :, 0], maskforCloud)
            plt.figure(figsize=[16,16])
            plt.imshow(masked_cloud, cmap='gray', vmin=0, vmax=1)
            plt.title('NDVI: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
            cb = plt.colorbar()
            cb.set_label("Cloud")
            plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/cloud_'+str(date))
            plt.clf()
            
        for v in range(vlen):
            for h in range(hlen):
                cloudAll[v,h,k]=arrCloud[v,h,0]
        
        ############################
        # Variables for Histogram  #
        ############################
        ndviRavel=arrNDVI[:,:,0].ravel()
        cloudRavel=arrCloud[:,:,0].ravel()
        
        cloud_mask=cloudRavel != 0
        ndviWithCloudMask=np.ma.masked_array(ndviRavel, cloud_mask)
        hist,edges=np.histogram(ndviWithCloudMask,bins=np.arange(-1.,1.01,.05))
        ndviHist[:,k]=hist
        ndviAvg[k]=np.average(ndviWithCloudMask) 
        if k>10:
            exit()
        ############################    
    #return ndviAll,cloudAll,ndviHist,ndviAvg,plotYear,k
    
    if makePlots:
        plt.clf()
        for v in range(vlen):
            for h in range(hlen):
                plt.figure(1)
                cloud_mask = cloudAll[v,h] != 0
                ndviWithCloudMask = np.ma.masked_array(ndviAll[v,h], cloud_mask)
                plt.plot(plotYear,ndviWithCloudMask,'.', color=(float(h)/float(hlen), 0.5, float(v)/float(vlen)))
                plt.ylim([-1.,1,])
                plt.xlabel('year')
                plt.ylabel('ndvi')
                plt.title('ndvi 2016 pixel '+str(v)+'_'+str(h))
                plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/ndvi_curve/cloud_masked_'+str(v)+'_'+str(h)+'.jpeg')
                plt.clf() 
    
    plt.clf()    
    plotYeartwoD=np.zeros(shape=(40,k))
    yvalue=np.zeros(shape=(40,k))
    for d in range(k):
        for v in range(40):
            plotYeartwoD[v,d]=plotYear[d]
            yvalue[v,d]=edges[v]
    
    
    x2=plotYear[0:k]
    ydata2=ndviAvg[0:k]
    yfit2=movingaverage(ydata2,8)
    
    plt.clf()
    plt.figure(1)
    plt.contourf(plotYeartwoD[:,0:k],yvalue[:,0:k],ndviHist[:,0:k],100,cmap=plt.cm.gist_stern_r,levels=np.arange(50))    
    plt.colorbar()
    plt.plot(x2[2:k-2],yfit2[2:k-2],'-k',linewidth=1)
    #plt.plot(plotYeartwoD[0,0:k],ndviAvg[0:k],'*',color='k')
    plt.title('NDVI 2016 '+str(lon)+'_'+str(lat))
    plt.xlabel('date')
    plt.ylabel('ndvi')
    plt.savefig(wd+'figures/summary/'+lonsave+'_'+latsave+'_heatmap')
    
    plt.clf()
    plt.plot(x2[2:k-2],yfit2[2:k-2],'-k',linewidth=1)
    plt.title('NDVI 2016 '+str(lon)+'_'+str(lat))
    plt.ylim(-1,1)
    plt.xlabel('date')
    plt.ylabel('ndvi')
    plt.savefig(wd+'figures/summary/'+lonsave+'_'+latsave+'_avgline')
    
    plt.clf()
    rollingmax=pd.rolling_max(ndviAvg[0:k],9)
    plt.plot(rollingmax[2:])
    plt.savefig(wd+'figures/summary/'+lonsave+'_'+latsave+'_rolling_max')
    
    
                                                                                                                                                                            
    
    images = dl.metadata.search(
        const_id=["CDL","CDL"],
        geom=dltile['geometry'],
        limit = 2000
        )
    
    n_images = len(images['features'])
    print('Number of image matches: %d' % n_images)
    k=n_images
    cdl = dl.raster.get_bands_by_constellation("CDL").keys()
    cdl1 = dl.raster.get_bands_by_constellation("CDL").keys()
    avail_bands = set(cdl).intersection(cdl1)
    print('Available bands: %s' % ', '.join([a for a in avail_bands]))
    
    band_info = dl.raster.get_bands_by_constellation("CDL")    
    
    try:
        valid_range = band_info['class']['valid_range']
        arr, meta = dl.raster.ndarray(
            'meta_2010_56m_cdls_v0',
            resolution=dltile['properties']['resolution'],
            bounds=dltile['properties']['outputBounds'],
            srs=dltile['properties']['cs_code'],
            bands=['class'],
            scales=[[valid_range[0], valid_range[1], None]],
            data_type='Float32'
            )
    except:
        print('class: %s could not be retreived' % scene)
    
    maskforCloud = arrCloud[:, :, 1] == 0    
    masked_cloud = np.ma.masked_array(arrCloud[:, :, 0], maskforCloud)
    plt.figure(figsize=[16,16])
    plt.imshow(arr, cmap='jet', vmin=0, vmax=255)
    #plt.title('NDVI: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
    cb = plt.colorbar()
    #cb.set_label("Cloud")
    plt.savefig(wd+'figures/tmp')
    plt.clf()
    
    ###############################################
    # Claculate Features     
    ############################################### 
    peakndvi=np.amax(rollingmax[2:])
    minndvi=np.amin(rollingmax[2:])
    variancendvi=Variance(rollingmax[2:])
    
    slope,bIntercept=np.polyfit(plotYear[2:],rollingmax[2:],1)
    yfit=slope*x+bIntercept
    
    ########################
    # Pickle variables     #
    ######################## 
    if not os.path.exists(r'/pickle_files/'+country+'/'+str(lon)+'_'+str(lat)):
        os.makedirs(r'/pickle_files/'+country+'/'+str(lon)+'_'+str(lat))
            
    pickle.dump(ndviAll,open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviAll','wb')) 
    pickle.dump(cloudAll,open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/cloudAll','wb'))
    pickle.dump(plotYear,open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/plotYear','wb'))
    pickle.dump(ndviHist,open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviHist','wb'))
    pickle.dump(ndviAvg,open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviAvg','wb'))
    pickle.dump(k,open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/k','wb'))
    ########################
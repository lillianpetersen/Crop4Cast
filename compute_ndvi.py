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
from celery import Celery
from scipy import ndimage

####################
# Function         #
####################
### Running mean/Moving average
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
    
def variance(x):   
    '''function to compute the variance (std dev squared)'''
    xAvg=np.mean(x)
    xOut=0.
    for k in range(len(x)):
        xOut=xOut+(x[k]-xAvg)**2
    xOut=xOut/(k+1)
    return xOut

def rolling_median(var,window):
    '''var: array-like. One dimension
    window: Must be odd'''
    n=len(var)
    halfW=int(window/2)
    med=np.zeros(shape=(var.shape))
    for j in range(halfW,n-halfW):
        med[j]=np.ma.median(var[j-halfW:j+halfW+1])
     
    for j in range(0,halfW):
        w=2*j+1
        med[j]=np.ma.median(var[j-w/2:j+w/2+1])
        i=n-j-1
        med[i]=np.ma.median(var[i-w/2:i+w/2+1])
    
    return med    
    
def mask_water(image):
    shape = image.shape
    length = image.size

    # reshape to linear
    x = image.reshape(length)

    # slice every 4th element
    y = x[0::4]

    # mask if less than 60 for NIR
    sixty = np.ones(len(y))*60
    z = y < sixty

    # multiply by 4
    oceanMask = np.repeat(z, 4)

    # apply mask to original array
    masked = np.ma.masked_array(x, oceanMask)
    b = np.ma.filled(masked, 0)

    # reshape
    c = b.reshape(shape)
    masked = masked.reshape(shape)
    oceanMask = oceanMask.reshape(shape)
    oceanMask = oceanMask[:,:,0]
    return c, oceanMask

def ltk_cloud_mask(X, get_rgb=False):
    #
    #   Modified Luo et al. (2008) LTK scheme (Oreopoulos et al. 2011)
    #   https://landsat.usgs.gov/documents/Oreopoulos_cloud.pdf
    #
    #    inputs:
    #    X       6-band landsat images : VIS/NIR/SWIR bands[1,2,3,4,5,7] in top-of-atmosphere reflectance
    #
    #    output:
    #    Y       byte-valued cloud/snow/water/shadow mask
    #    vals:   (based on official NASA LTK cloud mask labels)
    #    1       land
    #    2       snow
    #    3       water bodies
    #    4       clouds
    #    5       vegetation
    #

    L1 = X[:,:,0]
    L3 = X[:,:,1]
    L4 = X[:,:,2]
    L5 = X[:,:,3]

    Y = np.zeros(L1.shape, dtype='uint8')

    # stage 1 : non-vegetated land
    #
    indexA = (L1 < L3)
    indexA = np.logical_and(indexA, (L3 < L4))
    indexA = np.logical_and(indexA, (L4 < np.multiply(L5, 1.07)))
    indexA = np.logical_and(indexA, (L5 < 0.65))

    indexB = (np.multiply(L1, 0.8) < L3)
    indexB = np.logical_and(indexB, (L3 < np.multiply(L4, 0.8)))
    indexB = np.logical_and(indexB, (L4 < L5))
    indexB = np.logical_and(indexB, (L3 < 0.22))

    index = np.logical_and((Y == 0), np.logical_or(indexA, indexB))
    Y[index] = 1  # non-vegetated lands

    # stage 2 : snow/ice
    #
    indexA = (L3 >= 0.24)
    indexA = np.logical_and(indexA, (L5 < 0.16))
    indexA = np.logical_and(indexA, (L3 > L4))

    indexB = (L3 > 0.18)
    indexB = np.logical_and(indexB, (L3 < 0.24))
    indexB = np.logical_and(indexB, (L5 < np.subtract(L3, 0.08)))
    indexB = np.logical_and(indexB, (L3 > L4))

    index = np.logical_and((Y == 0), np.logical_or(indexA, indexB))
    Y[index] = 2  # snow/ice

    # stage 3 : water bodies
    #
    indexA = (L3 > L4)
    indexA = np.logical_and(indexA, (L3 > np.multiply(L5, 0.67)))
    indexA = np.logical_and(indexA, (L1 < 0.30))
    indexA = np.logical_and(indexA, (L3 < 0.20))

    indexB = (L3 > np.multiply(L4, 0.8))
    indexA = np.logical_and(indexA, (L3 > np.multiply(L5, 0.67)))
    indexB = np.logical_and(indexB, (L3 < 0.06))

    index = np.logical_and((Y == 0), np.logical_or(indexA, indexB))
    Y[index] = 3  # water bodies

    # stage 4 : clouds
    #
    index = np.logical_or((L1 > 0.28), (L3 > 0.30))
    index = np.logical_and(index, (L5 > 0.25))
    index = np.logical_and(index, (np.maximum(L1, L3) > np.multiply(L5, 0.90)))

    index = np.logical_and((Y == 0), index)
    Y[index] = 4  # clouds

    # stage 5 : vegetation
    #
    Y[(Y == 0)] = 5  # vegetation

    #
    if get_rgb:
        rgb = rgb_clouds(Y)
        return Y, rgb
    #
    globals().update(())
    return Y
####################        

#celery = Celery('compute_ndvi_forCloud', broker='redis://localhost:6379/0')
#
##wd='gs://lillian-bucket-storage/'
wd='/Users/lilllianpetersen/Google Drive/science_fair/'


# Celery task goes into start-up script

vlen=2016
hlen=2016
#vlen=994
#hlen=994
start='1990-01-01'
#start='2001-07-01'
end='2016-12-31'
nyears=16
country='Puerto_Rico'
makePlots=False
padding = 16
pixels = vlen+2*padding
res = 30

#vlen=100
#hlen=100
#padding=0
#pixels=vlen+2*padding
#    


#matches=dl.places.find('united-states_washington')
#matches=dl.places.find('north-america_united-states')
#matches=dl.places.find('south-america_brazil_rondonia')
#matches=dl.places.find('united-states_iowa')

matches=dl.places.find('puerto-rico_san-juan')
aoi = matches[0]
shape = dl.places.shape(aoi['slug'], geom='low')

#dltiles = dl.raster.dltiles_from_shape(res, pixels, padding, shape)
dltile=dl.raster.dltile_from_latlon(18.3,-66,res,vlen,padding)
#lonlist=np.zeros(shape=(len(dltiles['features'])))
#latlist=np.zeros(shape=(len(dltiles['features'])))

#for i in range(len(dltiles['features'])):
#    lonlist[i]=dltiles['features'][i]['geometry']['coordinates'][0][0][0]
#    latlist[i]=dltiles['features'][i]['geometry']['coordinates'][0][0][1]

#features=np.zeros(shape=(len(dltiles),nyears,pixels*pixels,6))
#target=np.zeros(shape=(len(dltiles),nyears,pixels*pixels))
features=np.zeros(shape=(len(dltile),nyears,pixels*pixels,6))
target=np.zeros(shape=(len(dltile),nyears,pixels*pixels))

#@celery.task  
def tile_function(dltile,makePlots=False):
   
    clas=["" for x in range(7)]
    clasLong=["" for x in range(255)]
    clasDict={}
    clasNumDict={}
    f=open(wd+'data/ground_data.txt')                                
    for line in f:
        tmp=line.split(',')
        clasNumLong=int(tmp[0])
        clasLong[clasNumLong]=tmp[1]
        clasNum=int(tmp[3])
        clas[clasNum]=tmp[2]
        
        clasDict[clasLong[clasNumLong]]=clas[clasNum]
        clasNumDict[clasNumLong]=clasNum
    
    lon=dltile['geometry']['coordinates'][0][0][0]
    lat=dltile['geometry']['coordinates'][0][0][1]
    globals().update(())
    print lon
    print lat
    latsave=str(lat)
    latsave=latsave.replace('.','-')
    lonsave=str(lon)
    lonsave=lonsave.replace('.','-')
    
    print '\n\n'
    #print 'dltile: '+str(tile)+' of '+str(len(dltiles['features']))
    

    #oceanMask=np.zeros(shape=(pixels,pixels))

    images = dl.metadata.search(
	products='landsat:LT05:PRE:TOAR',
        start_time=start,
        end_time=end,
        geom=dltile['geometry'],
        #cloud_fraction=0.8,
        limit = 2000
        )

    n_images = len(images['features'])
    print('Number of image matches: %d' % n_images)
    avail_bands = dl.raster.get_bands_by_constellation("L5").keys()
    print avail_bands 
    
    band_info=dl.metadata.bands(products='landsat:LT05:PRE:TOAR')

    dayOfYear=np.zeros(shape=(n_images))
    year=np.zeros(shape=(n_images),dtype=int)
    month=np.zeros(shape=(n_images),dtype=int)
    day=np.zeros(shape=(n_images),dtype=int)
    plotYear=np.zeros(shape=(n_images))
    xtime=[]
    i=-1
    globals().update(locals())
    for feature in images['features']:
        i+=1
        # get the scene id
        scene = feature['id']
            
        #xtime.append(str(images['features'][i]['id'][20:30]))
	xtime.append(str(images['features'][i]['properties']['acquired'][0:10]))
        date=xtime[i]
        year[i]=xtime[i][0:4]
        month[i]=xtime[i][5:7]
        day[i]=xtime[i][8:10]
        dayOfYear[i]=(float(month[i])-1)*30+float(day[i])
        plotYear[i]=year[i]+dayOfYear[i]/365.0
        
        
    indexSorted=np.argsort(plotYear)
    globals().update(locals())
    ####################
    # Define Variables #
    ####################
    print pixels
    ndviAll=-9999*np.ones(shape=(pixels,pixels,n_images))
    ndwiAll=np.zeros(shape=(pixels,pixels,n_images))
    Mask=np.ones(shape=(pixels,pixels,n_images),dtype=bool) 
    dayOfYear=np.zeros(shape=(n_images))
    year=np.zeros(shape=(n_images))
    month=np.zeros(shape=(n_images))
    day=np.zeros(shape=(n_images))
    plotYear=np.zeros(shape=(n_images))
    #ndviHist=np.zeros(shape=(40,n_images))
    #ndviAvg=np.zeros(shape=(n_images))
    #ndviMed=np.zeros(shape=(n_images))
    xtime=[]
    
    ndviHist=np.zeros(shape=(40,n_images))
    ndviAvg=np.zeros(shape=(n_images))
    ndviMed=np.zeros(shape=(n_images))
    ####################
    k=-1
    for j in range(len(indexSorted)):
        # get the scene id
        scene = images['features'][indexSorted[j]]['key']
       # ###############################################
       # # NDVI
       # ###############################################
       # # load the image data into a numpy array
       # try:
       #     default_range= band_info['default_range']
       #     physical_range = band_info['physical_range']
       #     arrNDVI, meta = dl.raster.ndarray(
       #         scene,
       #         resolution=dltile['properties']['resolution'],
       #         bounds=dltile['properties']['outputBounds'],
       #         srs=dltile['properties']['cs_code'],
       #         bands=['ndvi', 'alpha'],
       #         scales=[[default_range[0], default_range[1], physical_range[0], physical_range[1]]],
       #         #scales=[[0,16383,-1.0,1.0]],
       #         data_type='Float32'
       #         )
       # except:
       #     print('ndvi: %s could not be retreived' % scene)
       #     continue
       # globals().update(locals())
       
        
        #######################
        # Get cloud data      #
        #######################
        findCloud=-9999*np.ones(shape=(pixels,pixels,4)) 
        cloudMask=-9999*np.ones(shape=(pixels,pixels)) 
        try:
            default_range = band_info[2]['default_range']
            data_range = band_info[2]['physical_range']
            blue, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['blue', 'alpha'],
                scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
                data_type='Float32'
                )
        except:
            print('blue: %s could not be retreived' % scene)
            continue 

        try:
            default_range = band_info[1]['default_range']
            data_range = band_info[1]['physical_range']
            red, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['red', 'alpha'],
                scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
                data_type='Float32'
                )
        except:
            print('red: %s could not be retreived' % scene)
            continue 

        try:
            default_range = band_info[3]['default_range']
            data_range = band_info[3]['physical_range']
            nir, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['nir', 'alpha'],
                scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
                data_type='Float32'
                )
        except:
            print('nir: %s could not be retreived' % scene)
            continue
        
        try:
            default_range = band_info[5]['default_range']
            data_range = band_info[5]['physical_range']
            swir1, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['swir1', 'alpha'],
                scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
                data_type='Float32'
                )
        except:
            print('swir1: %s could not be retreived' % scene)
            continue 

	findCloud[:,:,0]=blue[:,:,0]
	findCloud[:,:,1]=red[:,:,0]
	findCloud[:,:,2]=nir[:,:,0]
	findCloud[:,:,3]=swir1[:,:,0]

	cloudMask[:,:]=ltk_cloud_mask(findCloud)
        globals().update(locals())

        #######################
        
        ###############################################
        # Test for bad days
        ############################################### 
    
        #take out days without data 
        #if arrCloud.shape == ()==True:
        #    continue
        #maskforCloud = arrCloud[:, :, 1] != 0 # False=Good, True=Bad
        if np.sum(cloudMask)==0:
            print 'continued'
            continue

        # take out days with too many clouds
        #cloudMask = arrCloud[:, :, 0] == 0 # True=good False=bad
        if np.sum(cloudMask)<0.15*(pixels*pixels):
            print 'clouds: continued'
            continue        
        k+=1
        
        ###############################################
        # time
        ############################################### 
        
        #xtime.append(str(images['features'][indexSorted[j]]['id'][20:30]))
	xtime.append(str(images['features'][k]['properties']['acquired'][0:10]))
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
        #cloudMask = arrCloud[:, :, 0] != 0 
        #cloudMask = arrCloud[:, :, 1] == 0 #for desert
        maskforAlpha = blue[:, :, 1] == 0 
        
	swap = {5:0,4:1,1:0,2:0,3:0,0:1}
	for v in range(pixels):
            for h in range(pixels):
		if cloudMask[v,h]==3 and v<600:
		    cloudMask[v,h]=1
		else:
                    cloudMask[v,h]=swap[cloudMask[v,h]]

        for v in range(pixels):
            for h in range(pixels):
                if cloudMask[v,h]==0 and maskforAlpha[v,h]==0: # and oceanMask[v,h]==0:
                    Mask[v,h,k]=0
        
	#Mask[:,:,k]=1-Mask[:,:,k]
	#Mask[:,:,k]=ndimage.binary_dilation(Mask[:,:,k],iterations=3)
	#Mask[:,:,k]=1-Mask[:,:,k]

	#cloudMask[:,:,k]=1-cloudMask[:,:,k]
	#cloudMask[:,:,k]=ndimage.binary_dilation(Mask[:,:,k],iterations=3)
	#cloudMask[:,:,k]=1-cloudMask[:,:,k]
       # if makePlots:
       #     

       #     masked_ndvi = np.ma.masked_array(arrNDVI[:, :, 0], Mask[:,:,k])
       #     plt.figure(figsize=[16,16])
       #     plt.imshow(masked_ndvi, cmap='jet', vmin=-.6, vmax=1)
       #     #plt.imshow(masked_ndvi, cmap='jet')#, vmin=0, vmax=65535)
       #     plt.title('NDVI: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
       #     cb = plt.colorbar()
       #     cb.set_label("NDVI")
       #     plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/ndvi_'+str(date)+'_'+str(k)+'.pdf')
       #     plt.clf() 
       #     
       # #for v in range(pixels):
       # #    for h in range(pixels):
       # #	if arrNDVI[	

       # ndviAll[:,:,k]=np.ma.masked_array(arrNDVI[:,:,0],Mask[:,:,k])
       # globals().update(locals())
	
        ###############################################
        # Cloud
        ###############################################
            
        if makePlots:
            plt.figure(figsize=[16,16])
            plt.imshow(cloudMask, cmap='gray', vmin=0, vmax=1)
            plt.title('Cloud: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
            cb = plt.colorbar()
            cb.set_label("Cloud")
            plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/cloud_'+str(date)+'_'+str(k)+'.pdf')
            plt.clf()
            
        ###############################################
        # NDWI
        ###############################################
        
        nirM=np.ma.masked_array(nir[:,:,0],Mask[:,:,k])
        
        try:
            default_range = band_info[4]['default_range']
            data_range = band_info[4]['physical_range']
            green, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['green', 'alpha'],
                scales=[[default_range[0], default_range[1], data_range[0], data_range[1]]],
                data_type='Float32'
                )
        except:
            print('green: %s could not be retreived' % scene)
            continue
          
        greenM=np.ma.masked_array(green[:,:,0],Mask[:,:,k])

        for v in range(pixels):
            for h in range(pixels):
        	ndwiAll[v,h] = (green[v,h,0]-nir[v,h,0])/(nir[v,h,0]+green[v,h,0]+1e-9)

        if makePlots:
            masked_ndwi = np.ma.masked_array(ndwiAll[:,:,k], Mask[:,:,k])
            plt.figure(figsize=[16,16])
            plt.imshow(masked_ndwi, cmap='jet', vmin=-1, vmax=1)
            plt.title('NDWI: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
            cb = plt.colorbar()
            cb.set_label("NDWI")
            plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/ndwi_'+str(date)+'_'+str(k)+'.pdf')
            plt.clf()
        
        globals().update(locals())
        
	###############################################
        # Visual
        ###############################################
        ids = [f['id'] for f in images['features']]

        arr, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['red', 'green', 'blue', 'alpha'],
                scales=[[0,4000], [0, 4000], [0, 4000], None],
                data_type='Byte',
                )

	if makePlots:
            plt.figure(figsize=[16,16])
            plt.imshow(arr)
            plt.title('visual')
            plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/visual_'+str(date)+'_'+str(k)+'.pdf')


	###############################################
        # Find number of standing water bodies 
        ###############################################
	

        ############################
        # Variables for Histogram  #
        ############################
        #ndviRavel=arrNDVI[:,:,0].ravel()
        ##        cloudRavel=arrCloud[:,:,0].ravel()
        #MaskRavel=Mask[:,:,0].ravel()
        #
        ##        cloud_mask=cloudRavel != 0
        #ndviWithMask=np.ma.masked_array(ndviRavel, MaskRavel)
        #hist,edges=np.histogram(ndviWithMask,bins=np.arange(-1.,1.01,.05))
        #ndviHist[:,k]=hist
        #ndviAvg[k]=np.mean(ndviWithMask)
        #ndviMed[k]=np.median(ndviWithMask)
        ############################    
    
    #return ndviAll,cloudAll,ndviHist,ndviAvg,plotYear,k

        #                ndwiAll[v,h,k] = np.clip(128.*(ndwiAll[v,h,k]+1), 0, 255).astype('uint8') shift range from [-1,1] to (0,255)
 
    
#    plt.clf()    
#    plotYeartwoD=np.zeros(shape=(40,k))
#    yvalue=np.zeros(shape=(40,k))
#    for d in range(k):
#        for v in range(40):
#            plotYeartwoD[v,d]=plotYear[d]
#            yvalue[v,d]=edges[v]
#    
#    
#    
#        rollingmed=rolling_median(ndviAvg[0:k],10)
#    rollingmed=rolling_median(ndviMed[0:k],10)
#    
#    x2=plotYear[0:k]
##    ydata2=ndviAvg[0:k]
#    ydata2=ndviMed[0:k]
#    yfit2=movingaverage(ydata2,16)
#    
#    plt.clf()
#    plt.figure(1)
#    plt.contourf(plotYeartwoD[:,0:k],yvalue[:,0:k],ndviHist[:,0:k],100,cmap=plt.cm.gist_stern_r,levels=np.arange(0,5000,10))    
#    plt.colorbar()
#    plt.plot(x2[2:k-2],yfit2[2:k-2],'.k',linewidth=1)
#    #plt.plot(plotYeartwoD[0,0:k],ndviAvg[0:k],'*',color='k')
#    plt.title('NDVI 2016 '+str(lon)+'_'+str(lat))
#    plt.xlabel('date')
#    plt.ylabel('ndvi')
#    plt.ylim(-1,1)
#    plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/_heatmap.pdf')
#    plt.clf()
#    
##        plt.plot(x2[2:k-2],yfit2[2:k-2],'.k',linewidth=1)
##        plt.title('NDVI 2016 '+str(lon)+'_'+str(lat))
##        plt.ylim(-1,1)
##        plt.xlabel('date')
##        plt.ylabel('ndvi')
##        plt.savefig(wd+'test_fig'+'_avgline.pdf')
##        plt.clf()
##        
##        plt.plot(rollingmed[2:])
##        plt.ylim(-1,1)
##        plt.savefig(wd+'test_fig'+'_rolling_max.pdf')
    globals().update(locals())

    ########################
    # Save variables       #
    ######################## 
    print lat,lon
    
    if not os.path.exists(r'../saved_vars/'+str(lon)+'_'+str(lat)):
        os.makedirs(r'../saved_vars/'+str(lon)+'_'+str(lat))
             
    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndwiAll',ndwiAll) 
    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/Mask',Mask)
#    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/oceanMask',oceanMask)
    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/plotYear',plotYear)
##    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviHist',ndviHist)
##    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviAvg',ndviAvg)
##    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviAvg',ndviMed)
    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/n_good_days',int(k))
    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/month',month)
    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/year',year)
##    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/edges',edges)
    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/arrClas',arrClas)
    np.save(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/ndviAll',ndviAll)
    
    
#for tile in range(len(dltiles['features'])):
#    dltile=dltiles['features'][tile]
#    print len(dltiles['features'])
tile_function(dltile,makePlots)   
    
    
#for i in range(len(dltiles['features'])):
#    ## Check in the bucket
#    ## gsutil ls
#    if not os.path.exists(r'../saved_vars/'+str(lonlist[i])+'_'+str(latlist[i])+'/ndviAll'):
#        dltile=dltiles['features'][i]
#        tile_function(dltile)
#        
        
        
        
        
        
        
        
        
        
        

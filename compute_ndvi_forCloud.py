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
    
####################        

wd='/Users/lilllianpetersen/Google Drive/science_fair/'

#lonAll=[-57.864114, -57.345734, -56.0979, -95.156364, -123.6585, -110.580639, 111.233326]
#latAll= [-13.458213, -12.748814,  -15.6014, 41.185114, 39.3592, 35.772751, 51.158285]
#lon=lonAll[4]
#lat=latAll[4]
#lon=-120.3631
#lat=38.4083
vlen=992
hlen=992
start='2016-01-01'
end='2016-12-31'
nyears=1
country='US'
makePlots=False
padding = 16
pixels = vlen+2*padding
res = 120.0

vlen=120
hlen=120
padding=4
pixels=vlen+2*padding

#compute_ndvi(lon,lat,pixels,start,end,country,makePlots)     

clas=["" for x in range(12)]
clasLong=["" for x in range(255)]
clasDict={}
clasNumDict={}
f=open('ground_data.txt')                                
for line in f:
    tmp=line.split(',')
    clasNumLong=int(tmp[0])
    clasLong[clasNumLong]=tmp[1]
    clasNum=int(tmp[3])
    clas[clasNum]=tmp[2]
    
    clasDict[clasLong[clasNumLong]]=clas[clasNum]
    clasNumDict[clasNumLong]=clasNum
    
    
"""function to compute ndvi and make summary plots
variables: lon, lat, pixels, start, end, country, makePlots
"""

#matches=dl.places.find('united-states_california')
#aoi = matches[0]
#shape = dl.places.shape(aoi['slug'], geom='low')

matches=dl.places.find('united-states_washington')
aoi = matches[0]
shape = dl.places.shape(aoi['slug'], geom='low')

#dltile = dl.raster.dltile_from_latlon(lat, lon, res, valid_pix, padding)
dltiles = dl.raster.dltiles_from_shape(res, vlen, padding, shape)

features=np.zeros(shape=(len(dltiles['features']),nyears,pixels*pixels,6))
target=np.zeros(shape=(len(dltiles['features']),nyears,pixels*pixels))

#features=pickle.load(open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/features','rd'))
#target=pickle.load(open(wd+'pickle_files/'+country+'/'+str(lon)+'_'+str(lat)+'/target','rd'))
exit()
for tile in range(len(dltiles['features'])):
    tile=900
    dltile=dltiles['features'][tile]
    lon=dltile['geometry']['coordinates'][0][0][0]
    lat=dltile['geometry']['coordinates'][0][0][1]
    
    latsave=str(lat)
    latsave=latsave.replace('.','-')
    lonsave=str(lat)
    lonsave=lonsave.replace('.','-')
    
    print '\n\n'
    print 'dltile: '+str(tile)+' of '+str(len(dltiles['features']))
    
    ###############################################
    # Find Ground Classification data    
    ###############################################                                                                                                                                                                         
    
    images = dl.metadata.search(
        const_id=["CDL","CDL"],
        geom=dltile['geometry'],
        limit = 2000
        )
    
    n_images = len(images['features'])
    #    print('Number of image matches: %d' % n_images)
    
    year=np.zeros(shape=(n_images),dtype='int')
    j=-1
    for feature in images['features']:
        j+=1
        scene=feature['id']
        
        year[j]=int(scene[14:18])
        if j==0:
            maxyear=year[j]
            maxj=j
            maxscene=scene
            continue
        
        if year[j]>maxyear:
            maxyear=year[j]
            maxj=j
            maxscene=scene
    
    
    cdl = dl.raster.get_bands_by_constellation("CDL").keys()
    cdl1 = dl.raster.get_bands_by_constellation("CDL").keys()
    avail_bands = set(cdl).intersection(cdl1)
    #    print('Available bands: %s' % ', '.join([a for a in avail_bands]))
    
    band_info = dl.raster.get_bands_by_constellation("CDL")    
    
    try:
        valid_range = band_info['class']['valid_range']
        arr, meta = dl.raster.ndarray(
            maxscene,
            resolution=dltile['properties']['resolution'],
            bounds=dltile['properties']['outputBounds'],
            srs=dltile['properties']['cs_code'],
            bands=['class'],
            scales=[[valid_range[0], valid_range[1]]],
            data_type='Float32'
            )
    except:
        print('class: %s could not be retreived' % maxscene)
    
    arr=arr.astype(int)
        
    arrClas=np.zeros(shape=(arr.shape))  
    for v in range(pixels):
        for h in range(pixels):
            arrClas[v,h]=clasNumDict[arr[v,h]]
    
    if makePlots:
        if not os.path.exists(r'../figures/'+country+'/'+str(lon)+'_'+str(lat)):
            os.makedirs(r'../figures/'+country+'/'+str(lon)+'_'+str(lat))
        plt.figure(figsize=[16,16])
        plt.imshow(arrClas, cmap='jet', vmin=0, vmax=11)
        #plt.title('NDVI: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
        plt.colorbar()
        #cb.set_label("Cloud")
        plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/groud_data_simple.pdf')
        plt.clf()
        
    if np.sum(arrClas)==0:
        print 'No Data: In the Ocean'
        continue
    
    oceanMask=np.zeros(shape=(arrClas.shape),dtype=bool)
    for v in range(pixels):
        for h in range(pixels):
            if arrClas[v,h]==0:
                oceanMask[v,h]=True
    ###############################################
    
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
    mo = dl.raster.get_bands_by_constellation("MO").keys()
    my = dl.raster.get_bands_by_constellation("MY").keys()
    avail_bands = set(mo).intersection(my)
    print('Available bands: %s' % ', '.join([a for a in avail_bands]))
    
    band_info = dl.raster.get_bands_by_constellation("MO")
    
    dayOfYear=np.zeros(shape=(n_images))
    year=np.zeros(shape=(n_images),dtype=int)
    month=np.zeros(shape=(n_images),dtype=int)
    day=np.zeros(shape=(n_images),dtype=int)
    plotYear=np.zeros(shape=(n_images))
    xtime=[]
    i=-1
    for feature in images['features']:
        i+=1
        # get the scene id
        scene = feature['id']
            
        xtime.append(str(images['features'][i]['id'][20:30]))
        date=xtime[i]
        year[i]=xtime[i][0:4]
        month[i]=xtime[i][5:7]
        day[i]=xtime[i][8:10]
        dayOfYear[i]=(float(month[i])-1)*30+float(day[i])
        plotYear[i]=year[i]+dayOfYear[i]/365.0
        
        
    indexSorted=np.argsort(plotYear)    
        
        
        
    ####################
    # Define Variables #
    ####################
    ndviAll=-9999*np.ones(shape=(pixels,pixels,n_images))
    ndwiAll=np.zeros(shape=(pixels,pixels,n_images))
#    cloudAll=-9999*np.ones(shape=(pixels,pixels,n_images)) 
    Mask=np.ones(shape=(pixels,pixels,n_images),dtype=bool) 
    dayOfYear=np.zeros(shape=(n_images))
    year=np.zeros(shape=(n_images))
    month=np.zeros(shape=(n_images))
    day=np.zeros(shape=(n_images))
    plotYear=np.zeros(shape=(n_images))
    ndviHist=np.zeros(shape=(40,n_images))
    ndviAvg=np.zeros(shape=(n_images))
    ndviMed=np.zeros(shape=(n_images))
    xtime=[]
    ####################
    k=-1
    
    for j in range(len(indexSorted)):
        # get the scene id
        scene = images['features'][indexSorted[j]]['key']
        
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
                scales=[[valid_range[0], valid_range[1], physical_range[0], physical_range[1]]],
                data_type='Float32'
                )
        except:
            print('ndvi: %s could not be retreived' % scene)
            continue
       
        ###############################################
        # Test for bad days
        ############################################### 
    
        #take out days without data 
        if arrNDVI.shape == ()==True:
            continue
        maskforNDVI = arrNDVI[:, :, 1] != 0 # False=Good, True=Bad
        if np.sum(maskforNDVI)==0:
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
                scales=[[valid_range[0], valid_range[1]]],
                data_type='Float32'
                )
        except:
            print('cloud: %s could not be retreived' % scene)
            continue 
        #######################
        
        #### Only for Desert ####
    #    for v in range(pixels):
    #        for h in range(pixels):
    #            arrCloud[:,:,0]=0
        #### Only for Desert ####
        
        # take out days with too many clouds
        maskforCloud = arrCloud[:, :, 0] == 0
        if np.sum(maskforCloud)<0.1*(pixels*pixels):
            print 'clouds: continued'
            continue        
        k+=1
        
        ###############################################
        # time
        ############################################### 
        
        xtime.append(str(images['features'][j]['id'][20:30]))
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
        maskforNDVI = arrNDVI[:, :, 1] == 0 
        
        for v in range(pixels):
            for h in range(pixels):
                if maskforCloud[v,h]==0 and maskforNDVI[v,h]==0 and oceanMask[v,h]==0:
                    Mask[v,h,k]=0
        
        if makePlots:
            
            if not os.path.exists(r'../figures/'+country+'/'+str(lon)+'_'+str(lat)):
                os.makedirs(r'../figures/'+country+'/'+str(lon)+'_'+str(lat))

            masked_ndvi = np.ma.masked_array(arrNDVI[:, :, 0], Mask[:,:,k])
            plt.figure(figsize=[16,16])
            plt.imshow(masked_ndvi, cmap='jet', vmin=-1, vmax=1)
            plt.title('NDVI: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
            cb = plt.colorbar()
            cb.set_label("NDVI")
            plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/ndvi_'+str(date)+'.pdf')
            plt.clf() 
            
        ndviAll[:,:,k]=np.ma.masked_array(arrNDVI[:,:,0],Mask[:,:,k])
        
        ###############################################
        # Cloud
        ###############################################
            
        if makePlots:
            #masked_cloud = np.ma.masked_array(arrCloud[:, :, 0], maskforCloud)
            masked_cloud = arrCloud[:, :, 0]
            plt.figure(figsize=[16,16])
            plt.imshow(masked_cloud, cmap='gray', vmin=0, vmax=1)
            plt.title('Cloud: '+str(lon)+'_'+str(lat)+', '+str(date), fontsize=20)
            cb = plt.colorbar()
            cb.set_label("Cloud")
            plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/cloud_'+str(date)+'.pdf')
            plt.clf()
            
#        for v in range(pixels):
#            for h in range(pixels):
#                cloudAll[v,h,k]=arrCloud[v,h,0]
        
        
        ###############################################
        # NDWI
        ###############################################
        
        try:
            valid_range = band_info['nir']['valid_range']
            physical_range = band_info['nir']['physical_range']
            nir, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['nir', 'alpha'],
                scales=[[valid_range[0], valid_range[1], physical_range[0], physical_range[1]]],
                data_type='Float32'
                )
        except:
            print('nir: %s could not be retreived' % scene)
            continue
        
        nirM=np.ma.masked_array(nir[:,:,0],Mask[:,:,k])
        
        try:
            valid_range = band_info['green']['valid_range']
            physical_range = band_info['green']['physical_range']
            green, meta = dl.raster.ndarray(
                scene,
                resolution=dltile['properties']['resolution'],
                bounds=dltile['properties']['outputBounds'],
                srs=dltile['properties']['cs_code'],
                bands=['green', 'alpha'],
                scales=[[valid_range[0], valid_range[1], physical_range[0], physical_range[1]]],
                data_type='Float32'
                )
        except:
            print('green: %s could not be retreived' % scene)
            continue
          
        greenM=np.ma.masked_array(green[:,:,0],Mask[:,:,k])
     
        for v in range(pixels):
            for h in range(pixels):
                if oceanMask[v,h]==True:
                    continue
                ndwiAll[v,h,k] = (greenM[v,h]-nirM[v,h])/(nirM[v,h]+greenM[v,h]+1e-9)
        #                ndwiAll[v,h,k] = np.clip(128.*(ndwiAll[v,h,k]+1), 0, 255).astype('uint8') shift range from [-1,1] to (0,255)
        '''
        ############################
        # Variables for Histogram  #
        ############################
        ndviRavel=arrNDVI[:,:,0].ravel()
#        cloudRavel=arrCloud[:,:,0].ravel()
        MaskRavel=Mask[:,:,0].ravel()
        
#        cloud_mask=cloudRavel != 0
        ndviWithMask=np.ma.masked_array(ndviRavel, MaskRavel)
        hist,edges=np.histogram(ndviWithMask,bins=np.arange(-1.,1.01,.05))
        ndviHist[:,k]=hist
        ndviAvg[k]=np.average(ndviWithMask)
        ndviMed[k]=np.median(ndviWithMask)
        ############################    
    
    #return ndviAll,cloudAll,ndviHist,ndviAvg,plotYear,k
    
    
    if makePlots:
        plt.clf()
        for v in range(pixels):
            for h in range(pixels):
                plt.figure(1)
                ndviWithMask = np.ma.masked_array(ndviAll[v,h], Mask)
                plt.plot(plotYear,ndviWithMask,'.', color=(float(h)/float(pixels), 0.5, float(v)/float(pixels)))
                plt.ylim([-1.,1,])
                plt.xlabel('year')
                plt.ylabel('ndvi')
                plt.title('ndvi 2016 pixel '+str(v)+'_'+str(h))
                plt.savefig(wd+'figures/'+country+'/'+str(lon)+'_'+str(lat)+'/cloud_masked_'+str(v)+'_'+str(h)+'.pdf')
                plt.clf() 
    
    plt.clf()    
    plotYeartwoD=np.zeros(shape=(40,k))
    yvalue=np.zeros(shape=(40,k))
    for d in range(k):
        for v in range(40):
            plotYeartwoD[v,d]=plotYear[d]
            yvalue[v,d]=edges[v]
    
    
    
    if makePlots:
#        rollingmed=rolling_median(ndviAvg[0:k],10)
        rollingmed=rolling_median(ndviMed[0:k],10)
        
        x2=plotYear[0:k]
    #    ydata2=ndviAvg[0:k]
        ydata2=ndviMed[0:k]
        yfit2=movingaverage(ydata2,16)
        
        plt.clf()
        plt.figure(1)
        plt.contourf(plotYeartwoD[:,0:k],yvalue[:,0:k],ndviHist[:,0:k],100,cmap=plt.cm.gist_stern_r,levels=np.arange(0,5000,10))    
        plt.colorbar()
        plt.plot(x2[2:k-2],yfit2[2:k-2],'.k',linewidth=1)
        #plt.plot(plotYeartwoD[0,0:k],ndviAvg[0:k],'*',color='k')
        plt.title('NDVI 2016 '+str(lon)+'_'+str(lat))
        plt.xlabel('date')
        plt.ylabel('ndvi')
        plt.ylim(-1,1)
        plt.savefig(wd+'figures/summary/'+lonsave+'_'+latsave+'_heatmap.pdf')
        plt.clf()
        
        plt.plot(x2[2:k-2],yfit2[2:k-2],'.k',linewidth=1)
        plt.title('NDVI 2016 '+str(lon)+'_'+str(lat))
        plt.ylim(-1,1)
        plt.xlabel('date')
        plt.ylabel('ndvi')
        plt.savefig(wd+'figures/summary/'+lonsave+'_'+latsave+'_avgline.pdf')
        plt.clf()
        
        plt.plot(rollingmed[2:])
        plt.ylim(-1,1)
        plt.savefig(wd+'figures/summary/'+lonsave+'_'+latsave+'_rolling_max.pdf')
    '''
    ########################
    # Save variables       #
    ######################## 
    if not os.path.exists(r'../saved_vars/'+country+'/'+str(lon)+'_'+str(lat)):
        os.makedirs(r'../saved_vars/'+country+'/'+str(lon)+'_'+str(lat))
            
    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviAll',ndviAll) 
    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/ndwiAll',ndwiAll) 
    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/Mask',Mask)
    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/plotYear',plotYear)
#    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviHist',ndviHist)
#    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviAvg',ndviAvg)
#    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/ndviAvg',ndviMed)
    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/n_good_days',k)
#    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/edges',edges)
    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/arrClas',arrClas)
    
    n_good_days=int(k)
    
    ###############################################
    # Claculate Features     
    ############################################### 
    
    ndviAllMask=np.ones(shape=(ndviAll.shape),dtype=bool)
    for v in range(pixels):
        for h in range(pixels):
            if oceanMask[v,h]==1:
                continue
            for t in range(n_good_days):
                if ndviAll[v,h,t]!=0 and ndviAll[v,h,t]>-1:
                    ndviAllMask[v,h,t]=False
    ndviAllM=np.ma.masked_array(ndviAll,Mask)
    ndwiAllM=np.ma.masked_array(ndwiAll,Mask)
    exit()
    ########################
    # Average NDVI Monthly #
    ######################## 
    ndviMonths=-9999.*np.ones(shape=(nyears,12,50))
    ndviMedMonths=-9999.*np.ones(shape=(nyears,pixels,pixels,12))
    ndvi90=np.zeros(shape=(nyears,pixels,pixels,12))
    ndvi10=np.zeros(shape=(nyears,pixels,pixels,12))
    
    # loop through years #
    for v in range(pixels):
        for h in range(pixels):  
            if oceanMask[v,h]==True:
                continue
            d=-1*np.ones(12,dtype=int)
            for t in range(n_good_days):
                if np.ma.is_masked(ndviAll[v,h,t])==False:
                    m=int(month[t])
                    y=year[t]-int(start[0:4])
                    d[m]+=1
                    ndviMonths[y,m,d[m]]=ndviAll[v,h,t]
        
            for y in range(nyears):
               for m in range(12):
                 ndviMedMonths[y,v,h,m]=np.median(ndviMonths[y,m,:int(d[m])])
                 ndvi90[y,v,h,m]=np.percentile(ndviMonths[m,:d[m]+1],90)
                 ndvi10[y,v,h,m]=np.percentile(ndviMonths[m,:d[m]+1],10)
    ###########################
    
    rollingmed_pix=np.zeros(shape=(pixels,pixels,k))
    for v in range(pixels):
        for h in range(pixels):
            if oceanMask[v,h]==False:
                rollingmed_pix[v,h,:]=rolling_median(ndviAll[v,h,:k],10)
    
    rollingmed_pix_mask=np.zeros(shape=(rollingmed_pix.shape),dtype=bool)
    for v in range(pixels):
        for h in range(pixels):
            if oceanMask[v,h]==True:
                rollingmed_pix_mask[v,h,:]=True
                continue
            for t in range(len(rollingmed_pix[0,0,:])):
                if math.isnan(rollingmed_pix[v,h,t])==True:
                    rollingmed_pix_mask[v,h,t]=True
    
    masked_rollingmed=np.ma.masked_array(rollingmed_pix,rollingmed_pix_mask)
    masked_plotYear=np.ma.masked_array(plotYear[0:k],rollingmed_pix_mask[0,0,:])
    
    
    parA=np.zeros(shape=(nyears,pixels,pixels))
    parB=np.zeros(shape=(nyears,pixels,pixels))
    parC=np.zeros(shape=(nyears,pixels,pixels))
    
    logm1=np.zeros(shape=(nyears,pixels,pixels))
    logb1=np.zeros(shape=(nyears,pixels,pixels))
    logm2=np.zeros(shape=(nyears,pixels,pixels))
    logb2=np.zeros(shape=(nyears,pixels,pixels))


    stdDev=np.zeros(shape=(nyears,pixels,pixels))
    
    ydata=np.zeros(shape=(nyears,pixels,pixels,n_good_days))
    x=np.zeros(shape=(nyears,n_good_days))
    ydataMask=np.zeros(shape=(nyears,pixels,pixels,n_good_days))
    
    i=np.zeros(shape=(nyears),dtype=int)
    for v in range(pixels):
        for h in range(pixels):
            if oceanMask[v,h]==True:
                continue
            i[:]=-1
            itmp=0
            for t in range(len(masked_rollingmed[0,0,:])):
                if np.is_masked(masked_rollingmed[v,h,t])==False:
                    y=year[t]-int(start[0:4])
                    i[y]+=1
                    ydata[y,v,h,i[y]]=masked_rollingmed[v,h,t]
                    x[y,i[y]]=masked_plotYear[t]
                    ydataMask[y,v,h,i[y]]=rollingmed_pix_mask[v,h,t]
    
    # if one year only
#    ydata[0,:,:,:]=masked_rollingmed[:,:,:]
#    x[0,:]=masked_plotYear[:]
#    
#    ydataM=np.ma.masked_array(ydata,rollingmed_pix_mask)
#    xM=np.ma.masked_array(x,rollingmed_pix_mask[0,0,:])
#    
#    ydata=np.ma.MaskedArray.filled(ydataM,fill_value=-9999.)
#    x=np.ma.MaskedArray.filled(xM,fill_value=-9999.)
#    
    ydata=np.ma.masked_array(ydata,ydataMask)
    x=np.ma.masked_array(x,ydataMask[0,0,:])
    
    
    for y in range(nyears):
        itmp=int(i[y])
#        itmp=n_good_days
        for v in range(pixels):
            for h in range(pixels):
                stdDev[y,v,h]=np.ma.std(ydata[y,v,h,:itmp])
        
                parA[y,v,h],parB[y,v,h],parC[y,v,h]=np.polyfit(x[y,:itmp],ydata[y,v,h,:itmp],2)
                logm1[y,v,h],logb1[y,v,h]=np.polyfit(x[y,:itmp/2], np.ma.log(ydata[y,v,h,:itmp/2]), 1)
                logm2[y,v,h],logb2[y,v,h]=np.polyfit(x[y,itmp/2:itmp], np.ma.log(ydata[y,v,h,itmp/2:itmp]), 1)
                
#                if makePlots:
#                    yfit=parA[y,v,h]*x[y,:itmp]**2+parB[y,v,h]*x[y,:itmp]+parC[y,v,h]
#                    plt.clf()
#                    plt.plot(x[y,:itmp],yfit,'.')
#                    plt.plot(x[y,:itmp],ydata[y,v,h,:itmp],'.')
#                    plt.ylim(0,1)
        #            plt.savefig(wd+'figures/parabola_'+lonsave+'_'+latsave+'_2015.pdf')
    
    
    stdDevR=np.reshape(stdDev,[nyears,pixels*pixels],order='C')
    parAr=np.reshape(parA,[nyears,pixels*pixels],order='C')
    parBr=np.reshape(parB,[nyears,pixels*pixels],order='C')
    parCr=np.reshape(parC,[nyears,pixels*pixels],order='C')
    
    ndvi90R=np.reshape(ndvi90,[nyears,pixels*pixels,12],order='C')
    ndvi10R=np.reshape(ndvi10,[nyears,pixels*pixels,12],order='C')
     
    arrClasR=np.reshape(arrClas,[pixels*pixels],order='C')
    
    for y in range(nyears):
        for p in range(pixels*pixels):
    #            print s
            features[tile,y,p,0]=ndvi90R[y,p,0]
            features[tile,y,p,1]=ndvi90R[y,p,1]
            features[tile,y,p,2]=ndvi90R[y,p,2]
            features[tile,y,p,3]=ndvi90R[y,p,3]
            features[tile,y,p,4]=ndvi90R[y,p,4]
            features[tile,y,p,5]=ndvi90R[y,p,5]
            features[tile,y,p,6]=ndvi90R[y,p,6]
            features[tile,y,p,7]=ndvi90R[y,p,7]
            features[tile,y,p,8]=ndvi90R[y,p,8]
            features[tile,y,p,9]=ndvi90R[y,p,9]
            features[tile,y,p,10]=ndvi90R[y,p,10]
            features[tile,y,p,11]=ndvi90R[y,p,11]
            
            features[tile,y,p,12]=ndvi10R[y,p,0]
            features[tile,y,p,13]=ndvi10R[y,p,1]
            features[tile,y,p,14]=ndvi10R[y,p,0]
            features[tile,y,p,15]=ndvi10R[y,p,3]
            features[tile,y,p,16]=ndvi10R[y,p,4]
            features[tile,y,p,17]=ndvi10R[y,p,5]
            features[tile,y,p,18]=ndvi10R[y,p,6]
            features[tile,y,p,19]=ndvi10R[y,p,7]
            features[tile,y,p,20]=ndvi10R[y,p,8]
            features[tile,y,p,21]=ndvi10R[y,p,9]
            features[tile,y,p,22]=ndvi10R[y,p,10]
            features[tile,y,p,23]=ndvi10R[y,p,11]
            
            features[tile,y,p,24]=stdDevR[y,p]
            features[tile,y,p,25]=parAr[y,p]
            features[tile,y,p,26]=parBr[y,p]
            features[tile,y,p,27]=parCr[y,p]
        
            target[tile,y,p]=arrClasR[p]
            
                
    
                
    ########################
    # Save variables       #
    ######################## 
    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/target',target)
    np.save(wd+'saved_vars/'+country+'/'+str(lon)+'_'+str(lat)+'/features',features)
    ########################

'''
targetR=np.reshape(target[:tile],tile*nyears*pixels*pixels)

sklearn.preprosessing.StandardScaler

for n in range(len(targetR)):
    if targetR[n]==1 or targetR[n]==2:
        targetR[n]=1
    elif targetR[n]==3 or targetR[n]==4:
        targetR[n]=2
    else:
        targetR[n]=targetR[n]-2
        
X=StandardScaler().fit_transform(featuresR)        


clf = svm.LinearSVC()
clf.fit()  
clf.predict()

'''
#ytst=10**(mtst*xtst+btst)+np.random.rand(180)*10
#
#m,b=np.polyfit(xtst, np.log10(ytst), 1)
#
#yfit=10**(m*xtst+b)
#
#plt.plot(xtst,ytst)
#plt.plot(xtst,yfit)
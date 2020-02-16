import pickle
from pylab import *
from sys import exit
import numpy as np
from datetime import datetime
from datetime import timedelta

latall=pickle.load(open('pickle_files/cmiplatall.p','rb'))
lonall=pickle.load(open('pickle_files/cmiplonall.p','rb'))

modelData85_tmax=pickle.load(open('pickle_files/modelData_all_tmax.p','rb'))
modelData85_tmin=pickle.load(open('pickle_files/modelData_all_tmin.p','rb'))

modelData45_tmax=pickle.load(open('pickle_files/modelData_all_tmax_rcp45.p','rb'))
modelData45_tmin=pickle.load(open('pickle_files/modelData_all_tmin_rcp45.p','rb'))

countyName=pickle.load(open('pickle_files/countyName.p','rb'))
startGrowingMon=pickle.load(open('pickle_files/startGrowingMon.p','rb'))
startGrowingDay=pickle.load(open('pickle_files/startGrowingDay.p','rb'))
endGrowingMon=pickle.load(open('pickle_files/endGrowingMon.p','rb'))
endGrowingDay=pickle.load(open('pickle_files/endGrowingDay.p','rb'))
cIndex=pickle.load(open('pickle_files/cIndex.p','rb'))
cIndexState=pickle.load(open('pickle_files/cIndexState.p','rb'))
T90all=pickle.load(open('pickle_files/T90all.p','rb'))
T10all=pickle.load(open('pickle_files/T10all.p','rb'))
slope=pickle.load(open('pickle_files/slope.p','rb'))
bIntercept=pickle.load(open('pickle_files/bIntercept.p','rb'))
presentGrowingCounties=pickle.load(open('pickle_files/presentGrowingCounties.p','rb'))
timeday=np.arange(42368,73049)
cropTitle=('corn','soybeans','rice')
runCrops=zeros(shape=(3))
icity=-1
nCities=3143 # number of cities  
nCrop=3
nyears=100
nScen=2
nPredictor=3

c=0

futureYield=-9999*ones(shape=(nCities,nPredictor,nyears,nScen,nCrop))

HeatWaves=-9999*ones(shape=(nScen,3143,100))
SummerAvg=-9999*ones(shape=(nScen,3143,100))
KDDays=-9999*ones(shape=(nScen,3143,100))

for i in range(nCities):
    icity+=1
    
    if icity!=604:
        continue
        
    if sum(presentGrowingCounties[icity])==0:
        continue
        
    doCorn=1
    doSoy=1
    doRice=1
    
    if presentGrowingCounties[icity,0]==0:
        doCorn=0
    if presentGrowingCounties[icity,1]==0:
        doSoy=0
    if presentGrowingCounties[icity,2]==0:
        doRice=0
        
    runCrops[0]=doCorn
    runCrops[1]=doSoy
    runCrops[2]=doRice
        
    lat=latall[icity]
    lon=lonall[icity]
    
    if int(lat)==-9999:
        c+=1
        continue
    
    city=countyName[icity].title()
    iBeg=16
    iEnd=100

    print icity,'of',nCities; sys.stdout.flush()

    dailyData=-9999*ones(shape=(nScen,2,nyears,12,31)) #v,y,m,d,scen
    
    for iday in range(len(timeday)):
        date_format = "%m/%d/%Y"
        d = timedelta(days=int(timeday[iday]))
        a = datetime.strptime('1/1/1900', date_format)
        date=a+d
        year=date.year
        month=date.month
        day=date.day
        
        y=year-2000
        y2=year-2016
        m=month-1
        d=day-1
           
        dailyData[0,0,y,m,d]=modelData85_tmax[d,m,y2,icity]
        dailyData[0,1,y,m,d]=modelData85_tmin[d,m,y2,icity]
        
        dailyData[1,0,y,m,d]=modelData45_tmax[d,m,y2,icity]
        dailyData[1,1,y,m,d]=modelData45_tmin[d,m,y2,icity]
    
    ###############################################
    # Make averages
    ###############################################
    mAvg=zeros(shape=(nScen,nyears,12))  #monthly average
    v=0
    
    ## compute yearly averages ##
    for scen in range(nScen):
        for y in range(iBeg,iEnd):
            for m in range(12):
                j=0
                for d in range(31):
                    if dailyData[scen,v,y,m,d]!=-9999:
                        mAvg[scen,y,m]+=dailyData[scen,v,y,m,d]
                        j+=1
                mAvg[scen,y,m]=mAvg[scen,y,m]/j
    
            
    yAvg=zeros(shape=(nScen,nyears))  #yearly average
    sAvg=zeros(shape=(nScen,nyears,4))  #seasonal average
    gAvg=zeros(shape=(nScen,nyears,nCrop)) #growing season average
    
    ## compute seasonal averages ##
    for scen in range(nScen):
        for y in range(iBeg,iEnd):
            yAvg[scen,y]=sum(mAvg[scen,y,:])/12
            
            sAvg[scen,y,0]=sum(mAvg[scen,y,2:5])/3 #spring MAM
            sAvg[scen,y,1]=sum(mAvg[scen,y,5:8])/3 #summer JJA
            sAvg[scen,y,2]=sum(mAvg[scen,y,8:11])/3 #fall SON
            sAvg[scen,y,3]=sum(mAvg[scen,y,0:2]+mAvg[scen,y,11])/3 #winter DJF
            
            SummerAvg[scen,icity,y]=sAvg[scen,y,1]
            
    ###############################################
    # Find Extremes
    ###############################################
    unsortedDall=zeros(shape=(nScen,4,366*315))
    # initialize variables for the extremes
    T90count=zeros(shape=(nScen,2,nyears))
    heatwavecount=zeros(shape=(nScen,2,nyears))
    seasonheatwavecount=zeros(shape=(nScen,2,nyears,nCrop))
    
    ## compute the extremes ##
    for scen in range(2):
        for v in range(1):
            i=3
            daysSinceHeatWave=0
            daysSinceColdWave=0
            for y in range(iBeg,iEnd):
                for m in range(12):
                    for d in range(31):
                        if dailyData[scen,v,y,m,d]>-100:
                            #unsortedDall[scen,v,i]=dailyData[scen,v,y,m,d]
                            unsortedDall[scen,v,i]=150
                            i+=1
                            daysSinceHeatWave+=1
                
                            if min(unsortedDall[scen,v,i-3:i])>=T90all[icity,v] and daysSinceHeatWave>2:
                                daysSinceHeatWave>2
                                heatwavecount[scen,v,y]+=1
                                daysSinceHeatWave=0
                                for cp in range(nCrop):
                                    if runCrops[cp]==0:
                                        continue
                                    if m>startGrowingMon[cIndexState[icity],cp] and m<endGrowingMon[cIndexState[icity],cp]:
                                        seasonheatwavecount[scen,v,y,cp]+=1
                                    
                                    elif m==startGrowingMon[cIndexState[icity],cp] and d>=startGrowingDay[cIndexState[icity],cp]:
                                        seasonheatwavecount[scen,v,y,cp]+=1
                                        
                                    elif m==endGrowingMon[cIndexState[icity],cp] and d<=endGrowingDay[cIndexState[icity],cp]:
                                        seasonheatwavecount[scen,v,y,cp]+=1
    
                HeatWaves[scen,icity,y]=heatwavecount[scen,0,y]
                                 
    ###############################################
    # Find KDD
    ###############################################  
    Tavg=zeros(shape=(nScen,nyears,12,31))
    KDD=zeros(shape=(nScen,nyears,nCrop))
    
    for scen in range(nScen):    
        for y in range(iBeg,iEnd):
            for m in range(12):
                for d in range(31):
                    if dailyData[scen,0,y,m,d]==-9999 or dailyData[scen,1,y,m,d]==-9999:
                        continue
                    dailytmin=dailyData[scen,1,y,m,d]
                    if dailytmin<50:
                        dailytmin=50
                    Tavg[scen,y,m,d]=(dailyData[scen,0,y,m,d]+dailytmin)/2
                    if Tavg[scen,y,m,d]<68:
                        continue
                    
                    for cp in range(nCrop):
                        if runCrops[cp]==0:
                                        continue
                        if m>startGrowingMon[cIndexState[icity],cp] and m<endGrowingMon[cIndexState[icity],cp]:
                            KDD[scen,y,cp]+=Tavg[scen,y,m,d]-68
                        
                        if m==startGrowingMon[cIndexState[icity],cp] and d>=startGrowingDay[cIndexState[icity],cp]:
                            KDD[scen,y,cp]+=Tavg[scen,y,m,d]-68  
                            
                        if m==endGrowingMon[cIndexState[icity],cp] and d<=endGrowingDay[cIndexState[icity],cp]:
                            KDD[scen,y,cp]+=Tavg[scen,y,m,d]-68 
    
            KDDays[scen,icity,y]=KDD[scen,y,0]
                           
    ###############################################
    # Predict Yields
    ###############################################
    ipredict=0    
    
    #print '\nKDD prediction:'
    for scen in range(nScen):
        #print 'scen=',scen
        for y in range(iBeg,iEnd):
            for cp in range(nCrop):
                if runCrops[cp]==0:
                    continue
                #print cropTitle[cp]   
                ## KDD prediction ##
                x=KDD[scen,:,cp]
                futureYield[icity,ipredict,y,scen,cp]=slope[2,icity,cp]*x[y]+bIntercept[2,icity,cp]
                futureYield[icity,ipredict,y,scen,cp]=round(futureYield[icity,ipredict,y,scen,cp],2)
                if -20<futureYield[icity,ipredict,y,scen,cp]<0:
                    futureYield[icity,ipredict,y,scen,cp]=0.01
                #print futureYield[icity,ipredict,y,scen,cp]
        #print '\n\n'
    ipredict+=1
    
    #print '\n\n\nSummer avg prediction:'
    for scen in range(nScen):
        #print 'scen=',scen
        for y in range(iBeg,iEnd):
            for cp in range(nCrop):
                if runCrops[cp]==0:
                    continue
                #print cropTitle[cp]
                    
                ## Summer avg prediction ##
                x=sAvg[scen,:,1]
                futureYield[icity,ipredict,y,scen,cp]=slope[3,icity,cp]*x[y]+bIntercept[3,icity,cp]
                futureYield[icity,ipredict,y,scen,cp]=round(futureYield[icity,ipredict,y,scen,cp],2)
                if -20<futureYield[icity,ipredict,y,scen,cp]<0:
                    futureYield[icity,ipredict,y,scen,cp]=0.01
                #print futureYield[icity,ipredict,y,scen,cp]
        #print '\n\n'
    ipredict+=1
    
    #print '\n\n\nHeat Wave prediction:'
    for scen in range(nScen):
        #print 'scen=',scen
        for y in range(iBeg,iEnd):
            for cp in range(nCrop):
                if runCrops[cp]==0:
                    continue
                #print cropTitle[cp]
                    
                ## Heat Wave prediction ##
                x=seasonheatwavecount[scen,0,:,cp]
                futureYield[icity,ipredict,y,scen,cp]=slope[4,icity,cp]*x[y]+bIntercept[4,icity,cp]
                futureYield[icity,ipredict,y,scen,cp]=round(futureYield[icity,ipredict,y,scen,cp],2)
                if -20<futureYield[icity,ipredict,y,scen,cp]<0:
                    futureYield[icity,ipredict,y,scen,cp]=0.01
                #print futureYield[icity,ipredict,y,scen,cp]
        #print '\n\n'

        
pickle.dump(futureYield,open('pickle_files/futureYield170105.p','wb'))  
pickle.dump(SummerAvg,open('pickle_files/SummerAvgFuture.p','wb'))
pickle.dump(HeatWaves,open('pickle_files/HeatWavesFuture.p','wb'))
pickle.dump(KDDays,open('pickle_files/KDDFuture.p','wb'))    







#imshow(tmax2[0,:,:],origin='lower',extent=[lon[0],lon[540],lat[0],lat[382]],vmin=255,vmax=300)
#colorbar()
#savefig('final_figures/model_data')
#show()
 

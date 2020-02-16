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
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#from celery import Celery


#celery = Celery('compute_ndvi', broker='redis://localhost:6379/0')

#wd='gs://lillian-bucket-storage/'
wd='/Users/lilllianpetersen/Google Drive/science_fair/'


vlen=992
hlen=992
start='2015-01-01'
end='2016-12-31'
nyears=2
#start='2016-01-01'
#end='2016-12-31'
#nyears=1
country='US'
makePlots=False
padding = 16
pixels = vlen+2*padding
res = 120.0

vlen=100
hlen=100
padding=0
pixels=vlen+2*padding
    

matches=dl.places.find('united-states_iowa')
#matches=dl.places.find('united-states_washington')
aoi = matches[0]
shape = dl.places.shape(aoi['slug'], geom='low')

clas=["" for x in range(12)]
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
    

dltiles = dl.raster.dltiles_from_shape(res, vlen, padding, shape)

tile=len(dltiles['features'])-1
tile=4
lon=dltiles['features'][tile]['geometry']['coordinates'][0][0][0]
lat=dltiles['features'][tile]['geometry']['coordinates'][0][0][1]

latsave=str(lat)
latsave=latsave.replace('.','-')
lonsave=str(lat)
lonsave=lonsave.replace('.','-')

features=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/features.npy')
target=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/target.npy')
#features=features[tile]
#target=target[tile]
featuresR=features.reshape(nyears*features.shape[1],30)
targetR=target.reshape(nyears*features.shape[1])
#clas=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/clas.npy')
#featuresR=features.reshape(nyears*features.shape[2],30)
#targetR=target.reshape(nyears*features.shape[2])

# break the data up into different classes
classes=-9999*np.ones(shape=(10,len(featuresR),30))
avg_classes=-9999*np.ones(shape=(10,30))
pix=-1*np.ones(shape=(10),dtype=int)
for i in range(len(featuresR)):
    cls=int(targetR[i])
    pix[cls]+=1
    classes[cls,pix[cls],:]=featuresR[i,:]

# get an even number of each class to test on
#pxNum=289/2
pxNum=13236/2
#clsnum=9
clsnum=6
predict_data=np.zeros(shape=(pxNum*clsnum,30))
y_true=np.zeros(shape=(pxNum*clsnum))
X=np.zeros(shape=(pxNum*clsnum,30))
y=np.zeros(shape=(pxNum*clsnum))
i=-1
#for cls in range(9):
#for cls in [1,4,6,7,8]:
for cls in [1,4,5,6,7,8]:
    for px in range(pxNum):
        i+=1
        
        X[i,:]=classes[cls,px,:]
        y[i]=cls
i=-1
pxcls=np.zeros(shape=(9))
#for cls in range(9):
#for cls in [1,4,6,7,8]:
for cls in [1,4,5,6,7,8]:
    for px in range(pxNum):
        i+=1
        predict_data[i,:]=classes[cls,px+pxNum,:]
        y_true[i]=cls
        

#clf = RandomForestClassifier(n_estimators=15, max_features=5, n_jobs=-1)
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X,y)
y_pred=clf.predict(predict_data)

clas_labels=[]
clas_labels.append(clas[1])
clas_labels.append(clas[4])
clas_labels.append(clas[5])
clas_labels.append(clas[6])
clas_labels.append(clas[7])
clas_labels.append(clas[8])

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(1,figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)


# Plot normalized confusion matrix
plt.figure(1,figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=clas_labels, normalize=True,
                      title='Normalized confusion matrix')
exit()
#
plt.savefig(wd+'figures/US/'+str(lon)+'_'+str(lat)+'/conf_matrix_normalized.pdf')
#plt.show()

for cls in range(10):
    for ftr in range(30):
        avg_classes[cls,ftr]=np.mean(classes[cls,:pix[cls],ftr])

#fHist=np.zeros(shape=(40,np.amax(px)))
#for cls in range(10):
#    cls=5
#    for i in range(np.amax(px)):
#        hist,edges=np.histogram(classes[cls,i,:24],bins=np.arange(-1.,1.01,.05))
#        fHist[:,i]=hist[:pix[cls]]
#    plt.clf()
#    plt.figure(1)
#    plt.contourf(np.arange(pix[cls]),edges[:40],fHist,100,cmap=plt.cm.gist_stern_r,levels=np.arange(0,5000,10))    
#    plt.colorbar()
 
    

plt.clf()
plt.plot(avg_classes[5,12:24],'--b',label=clas[5])
plt.plot(avg_classes[5,:12],'--b')
plt.plot(avg_classes[7,:12],'--g',label=clas[7])
plt.plot(avg_classes[7,12:24],'--g')
plt.plot(avg_classes[6,:12],'--r',label=clas[6])
plt.plot(avg_classes[6,12:24],'--r')
plt.plot(avg_classes[1,:12],'--y',label=clas[1])
plt.plot(avg_classes[1,12:24],'--y')
plt.plot(avg_classes[4,:12],'--m',label=clas[4])
plt.plot(avg_classes[4,12:24],'--m')
plt.plot(avg_classes[8,:12],'--k',label=clas[8])
plt.plot(avg_classes[8,12:24],'--k')
plt.axis([0,12,0,1])
plt.legend()
plt.savefig(wd+'figures/US/'+str(lon)+'_'+str(lat)+'/different_land_types.pdf')













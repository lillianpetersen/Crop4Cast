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
start='2016-01-01'
startyear=2016
end='2016-12-31'
nyears=1
country='US'
makePlots=False
padding = 16
pixels = vlen+2*padding
res = 120.0

matches=dl.places.find('united-states_washington')
aoi = matches[0]
shape = dl.places.shape(aoi['slug'], geom='low')

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
clas=np.load(wd+'saved_vars/'+str(lon)+'_'+str(lat)+'/clas.npy')
featuresR=features.reshape(1*1048576,30)
targetR=target.reshape(1*1048576)

#X=featuresR[:1000000]
#Y=targetR[:1000000]

classes=-9999*np.zeros(shape=(10,len(featuresR),30))
avg_classes=np.zeros(shape=(10,30))
px=-1*np.ones(shape=(10),dtype=int)
for i in range(len(featuresR)):
    cls=int(targetR[i])
    px[cls]+=1
    classes[cls,px,:]=featuresR[i,:]

pxNum=177
learn_data=np.zeros(shape=(pxNum*9+1,30))
y_true=np.zeros(shape=(pxNum*9+1))
X=np.zeros(shape=(pxNum*9+1,30))
y=np.zeros(shape=(pxNum*9+1))
i=-1
for cls in range(9):
    for px in range(pxNum):
        i+=1
        
        X[i,:]=classes[cls,px,:]
        y[i]=cls
i=-1
for cls in range(9):
    for px in range(100):
        i+=1
        learn_data[i,:]=classes[cls,px+pxNum,:]
        y_true[i]=cls
        

clf = RandomForestClassifier()
clf.fit(X,y)
y_pred=clf.predict(learn_data)

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
plot_confusion_matrix(cnf_matrix, classes=clas, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig(wd+'/figures/conf_matrix_unnormalized.pdf')
plt.show()

for cls in range(10):
    for ftr in range(30):
        avg_classes[cls,ftr]=np.mean(classes[cls,:px[cls],ftr])
exit()
fHist=np.zeros(shape=(40,np.amax(px)))
for cls in range(10):
    cls=5
    for i in range(np.amax(px)):
        hist,edges=np.histogram(classes[cls,i,:24],bins=np.arange(-1.,1.01,.05))
        fHist[:,i]=hist[:px[cls]]
    plt.clf()
    plt.figure(1)
    plt.contourf(np.arange(px[cls]),edges[:40],fHist,100,cmap=plt.cm.gist_stern_r,levels=np.arange(0,5000,10))    
    plt.colorbar()
    exit()















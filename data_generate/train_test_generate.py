import random
import numpy as np
 
getTrain = []
getTest = []
getVal = []
 
arrTrain = []
arrTest = []
arrVal = []
 
arr = np.arange(1,3559)
random.shuffle(arr)
# training set
getTrain = arr[0:2490]
getTrain.sort()
 
f=open('VOCdevkit/VOC2007/ImageSets/train.txt','w')
for i in getTrain:
    f.write('{:06d}\n'.format(i))
# testing set
getTest = arr[2490:3024]
getTest.sort()
 
f=open('VOCdevkit/VOC2007/ImageSets/test.txt','w')
for i in getTest:
    f.write('{:06d}\n'.format(i))
# validation set
getVal = arr[3024:3558]
getVal.sort()
 
f=open('VOCdevkit/VOC2007/ImageSets/val.txt','w')
for i in getVal:
    f.write('{:06d}\n'.format(i))
# train + val
getTrainVal = np.concatenate((getTrain,getVal),axis=None)
getTrainVal.sort()
 
f=open('VOCdevkit/VOC2007/ImageSets/trainval.txt','w')
for i in getTrainVal:
    f.write('{:06d}\n'.format(i))

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  
import matplotlib.image as mpimg 
import imageio
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize

"""
import scipy.misc
scipy.misc.toimage(image_array).save('outfile.jpg')
"""

import cv2

"""
x= (x_ano-x_offset)/x_ElementSpacing 
y= (y_ano-y_offset)/y_ElementSpacing 
z= (z_ano-z_offset)/z_ElementSpacing 
"""

class CTImage(object):
    def __init__(self, x_offset, y_offset, z_offset, x_ElementSpacing, y_ElementSpacing, z_ElementSpacing):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.x_ElementSpacing = x_ElementSpacing
        self.y_ElementSpacing = y_ElementSpacing
        self.z_ElementSpacing = z_ElementSpacing
        self.ElementSpacing = np.array([x_ElementSpacing, y_ElementSpacing, z_ElementSpacing])


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        offset = [k for k in contents if k.startswith('Offset')][0]
        EleSpacing = [k for k in contents if k.startswith('ElementSpacing')][0]
        
        # get data from files
        offArr = np.array(offset.split(' = ')[1].split(' ')).astype('float')
        eleArr = np.array(EleSpacing.split(' = ')[1].split(' ')).astype('float')
        CT = CTImage(offArr[0],offArr[1],offArr[2],eleArr[0],eleArr[1],eleArr[2])
        transform = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transform = np.round(transform)
        if np.any(transform != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])): # Check if image is flipped or not
            isflip = True
        else:
            isflip = False
    itkimage = sitk.ReadImage(filename)
    numpyimage = sitk.GetArrayFromImage(itkimage)
    if(isflip == True):
        numpyimage = numpyimage[:,::-1,::-1] 
    return (numpyimage,CT,isflip)

def truncate_hu(image_array):
    image_array[image_array > 400] = 400
    image_array[image_array <-1000] = -1000
    

def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array - min)/(max - min)*255
    #image_array = image_array.astype(int)#整型
    image_array = np.round(image_array)
    return image_array 
# lung masking
def findMaxRegion(img):
    img = img.astype(np.uint8)
    num, labels = cv2.connectedComponents(img, connectivity = 4)        
    getLabel = [0]    
    for i in range(1, num):
        getLabel.append(np.sum(labels==i))           
    maxNum = 0
    for i in getLabel:
        if i > maxNum:
            maxNum = i
    getIndex = []
    for i in range(num):
        if getLabel[i] > maxNum/2:
            getIndex.append(i)
    maskLabelInedx = [np.where(labels == x) for x in getIndex ]        
    maskLabel = np.zeros(labels.shape, dtype = int)  
    for i in range(len(maskLabelInedx)):
        maskLabel[maskLabelInedx[i]] = 1 
    return maskLabel

def fill_color_demo(image):
    copyImage = image.copy()
    h, w = copyImage.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8) 
    cv2.floodFill(copyImage, mask, (20, 20), (0, 125, 125), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    #cv2.imshow("fill_color_demo", copyImage)


def getLungMask(imNoudle):
    img = imNoudle
    #Standardize the pixel values
    mean = np.mean(imNoudle)
    std = np.std(imNoudle)
    imNoudle = imNoudle-mean
    imNoudle = imNoudle/std
    '''
    # nodule distribution
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
    ax1.imshow(imNoudle,cmap='gray')
    plt.hist(imNoudle.flatten(),bins=200)
    plt.show()
    cv2.imshow("1",imNoudle)
    '''
    # nomalize lung part
    middle = imNoudle[100:400,100:400] 
    mean = np.mean(middle)  
    #using KMean method seperate foreground and background
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(imNoudle<threshold,1.0,0.0)  # threshold the image   
    #cv2.imshow("thresh_img",thresh_img) 
    image_array = thresh_img
    #plt.imshow(image_array,cmap='gray') 
    #plt.show()           
    # erosion and dilation   
    erosion = cv2.erode(thresh_img, np.ones((4, 4), np.uint8))  
    dilation = cv2.dilate(erosion, np.ones((10, 10), np.uint8))  
    #dilation = dilation * np.ones(dilation.shape, dtype = np.uint8)    
    #cv2.imshow("after change",dilation)
    #eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    #dilation = morphology.dilation(eroded,np.ones([10,10]))
 
    labels = measure.label(dilation)
    '''  
    fig,ax = plt.subplots(2,2,figsize=[8,8])
    ax[0,0].imshow(thresh_img,cmap='gray')  
    ax[0,1].imshow(erosion,cmap='gray') 
    ax[1,0].imshow(dilation,cmap='gray')  
    ax[1,1].imshow(labels) 
    plt.show()    
    '''
    # only take unique value, drop all duplicated value
    label_vals = np.unique(labels)
    # measuring connect regions
    regions = measure.regionprops(labels)
    good_labels = []
    """
    area	int	   pixels in area
    bbox	tuple  (min_row, min_col, max_row, max_col)
    centroid	array
    convex_area	int	
    convex_image	　　
    coords	ndarray	
    Eccentricity 	float	
    label	int	
    """
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    # lung mask
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = mask.astype(np.uint8)
    #mask = cv2.erode(mask, np.ones((10, 10)))# erosion will create some holes
    mask = cv2.dilate(mask, np.ones((10, 10)))# dilation fills the holes
    #mask = morphology.erosion(mask,np.ones([10,10]))#erosion remove some edges
    
    maskLabel = findMaxRegion(mask)
    '''
    fig,ax = plt.subplots(2,2,figsize=[10,10])
    ax[0,0].imshow(img)  
    ax[0,1].imshow(img,cmap='gray')  
    ax[1,0].imshow(maskLabel,cmap='gray')  
    ax[1,1].imshow(img*maskLabel,cmap='gray')  
    plt.show()
    '''
    
    
    #cv2.imshow("after mask imNoudle",(img * mask).astype(np.uint8))
    thresh_img_mask = np.ones(imNoudle.shape, dtype = np.uint8) * mask * 255
    thresh_img_mask = thresh_img_mask.astype(np.uint8)
    fill_color_demo(thresh_img_mask)
    #cv2.imshow("1thresh_img_mask",thresh_img_mask)
    """
    """
    maskLabel = findMaxRegion(mask)
    img = img * maskLabel
    img = img.astype(np.uint8)
    
    thresh_img_mask = np.ones(imNoudle.shape, dtype = np.uint8) * maskLabel * 255
    thresh_img_mask = thresh_img_mask.astype(np.uint8)
    imageio.imwrite('thresh_img_mask2.jpg',thresh_img_mask)
    #cv2.imshow("2thresh_img_mask",thresh_img_mask)
    #cv2.imshow("end",img)
    #cv2.waitKey(0)
    """
    """
    return img, maskLabel

   
    
if __name__ == "__main__":
    
    case_path = 'data/subset8/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd' 
    numpyimage, _,_ = load_itk_image(case_path)
    print("numpyimage:\n",type(numpyimage))
    print("numpyimage.shape: ",numpyimage.shape)
    imNoudle = numpyimage.transpose(1,2,0)[:,:,170]
    
    truncate_hu(numpyimage)
    image_array = normalazation(numpyimage)
    imNoudle = image_array.transpose(1,2,0)[:,:,193]
    img = imNoudle
    img2 = imNoudle
    imNoudle = imNoudle.astype(np.uint8)
    #cv2.imshow("begin imNoudle",imNoudle)
    cv2.imwrite("imNoudle.jpg",imNoudle)
    print("image_array.shape",image_array.shape)
    img, mask = getLungMask(img)
    #cv2.imshow("11",img)
    #cv2.waitKey(0)





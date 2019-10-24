
from xml.etree.ElementTree import Element,ElementTree,tostring
import json,csv
import pandas as pd
from getImg import load_itk_image, truncate_hu, normalazation, getLungMask
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np

## Create folders###########################################
try:
    os.makedirs("VOCdevkit/VOC2007/Annotations")
    os.makedirs("VOCdevkit/VOC2007/JPEGImages")
    os.makedirs("VOCdevkit/VOC2007/LabelImages")
    os.makedirs("VOCdevkit/VOC2007/ImageSets")
except:
    print('Folders already created, beginning next step.......\n')



def csvtoxml(fname):
    with open(fname,'r') as f:
        reader=csv.reader(f)
        header=next(reader)
        root=Element('Daaa')
        print('root',len(root))
        for row in reader:
            erow=Element('Row')
            root.append(erow)
            for tag,text in zip(header,row):
                e=Element(tag)
                e.text=text
                erow.append(e)
    beatau(root)
    return ElementTree(root)
 
 
 
def beatau(e,level=0):
    if len(e)>0:
        e.text='\n'+'\t'*(level+1)
        for child in e:
           beatau(child,level+1)
        child.tail=child.tail[:-1]
    e.tail='\n' + '\t'*level

# Global to Voxel coordinate
def worldToVoxelCoord(worldCoord, offset, EleSpacing):

    stretchedVoxelCoord = np.absolute(worldCoord - offset)
    voxelCoord = stretchedVoxelCoord / EleSpacing

    return voxelCoord
# Voxel to Global coordinate
def VoxelToWorldCoord(voxelCoord, origin, spacing):

    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin

    return worldCoord

# Convert to VOC annotation "xml" format
def ToXml(name, x, y, w, h):
    root = Element('annotation')
    erow1 = Element('folder')
    erow1.text= "VOC"
    
    
    erow2 = Element('filename')
    erow2.text= str(name)
    
    erow3 = Element('size')
    erow31 = Element('width')
    erow31.text = "512"
    erow32 = Element('height')
    erow32.text = "512"
    erow33 = Element('depth') # RGB 
    erow33.text = "3" 
    erow3.append(erow31)
    erow3.append(erow32)
    erow3.append(erow33)
    
    erow4 = Element('object')
    erow41 = Element('name')
    erow41.text = 'nodule'
    erow43 = Element('difficult')
    erow43.text = "0"
    erow42 = Element('bndbox')
    erow4.append(erow41)
    erow4.append(erow43)
    erow4.append(erow42)
    erow421 = Element('xmin')
    erow421.text = str(x - np.round(w/2).astype(int))
    erow422 = Element('ymin')
    erow422.text = str(y - np.round(h/2).astype(int))
    erow423 = Element('xmax')
    erow423.text = str(x + np.round(w/2).astype(int))
    erow424 = Element('ymax')
    erow424.text = str(y + np.round(h/2).astype(int))
    erow42.append(erow421)
    erow42.append(erow422)
    erow42.append(erow423)
    erow42.append(erow424)
    
    root.append(erow1)
    root.append(erow2)
    root.append(erow3)            
    root.append(erow4)  
    beatau(root)      
    
    return ElementTree(root)
# file path searching 
def search(path=".", name="",fileDir = []):
    
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            search(item_path, name)
        elif os.path.isfile(item_path):            
            if name in item:
                fileDir.append(item_path)
                #print("fileDir:",fileDir)
                
    return fileDir


if __name__ == "__main__":   
    
    # Labels
    fname = 'CSVFILES/annotations.csv'
    data = pd.read_csv(fname)    
    # look through all files
    nameLast = []
    namePre = []
    fileDir = []
    count = 0 
    for i in range(len(data)):
        # read '.mhd' files 
        print("{}   /total {}   ".format(i,len(data)))          
        namePre = data['seriesuid'].loc[i]
        if namePre != nameLast :# if different name with previous one, clear cache and read data
            fileDir.clear()          
            fileDir = search(path=r"data/", name = namePre)
            print(fileDir)   
            for file in fileDir:
                if '.mhd' in file:
                    getDir = file
                    break

        # get coordinates
        x_ano = data['coordX'].loc[i]
        y_ano = data['coordY'].loc[i]
        z_ano = data['coordZ'].loc[i]
        r = data['diameter_mm'].loc[i]

        numpyimage, CT, isflip = load_itk_image(getDir)
        truncate_hu(numpyimage) # get HU values 
        image_array = normalazation(numpyimage) # normalize to 0~256
        # coordinate calculation
        """
        print("z_anp    :",z_ano)
        print("z_offset :",CT.z_offset)
        print("z_anp - CT.z_offset:",z_ano - CT.z_offset)
        """
        x = np.round(worldToVoxelCoord(x_ano, CT.x_offset, CT.x_ElementSpacing)).astype(int)
        y = np.round(worldToVoxelCoord(y_ano, CT.y_offset, CT.y_ElementSpacing)).astype(int)
        z = np.round(worldToVoxelCoord(z_ano, CT.z_offset, CT.z_ElementSpacing)).astype(int)
        w = np.round(r/CT.x_ElementSpacing).astype(int)
        h = np.round(r/CT.y_ElementSpacing).astype(int)
        #print("x    :",x)
        #print("y    :",y)
        #print("z    :",z)
        
        # take 3 images for each nodule (center, upper closest, lower closest)
        # transpose original (z,x,y) format to (x,y,z)
        imgLabel1 = image_array.transpose(1,2,0)[:,:,z - 1] 
        imgLabel2 = image_array.transpose(1,2,0)[:,:,z] 
        imgLabel3 = image_array.transpose(1,2,0)[:,:,z + 1] 
        # Convert to RGB format
        # Lung truncating
        img, mask = getLungMask(imgLabel1)
        im1 = Image.fromarray(img)
        im1 = im1.convert("RGB")      
        #im1.save('VOCdevkit/VOC2007/JPEGImages/{:06d}.jpg'.format(count+1))
        count = count + 1
        img, mask = getLungMask(imgLabel2)
        im2 = Image.fromarray(img)
        im2 = im2.convert("RGB")      
        #im2.save('VOCdevkit/VOC2007/JPEGImages/{:06d}.jpg'.format(count+1))
        count = count + 1
        img, mask = getLungMask(imgLabel3)
        im3 = Image.fromarray(img)
        im3 = im3.convert("RGB")      
        #im3.save('VOCdevkit/VOC2007/JPEGImages/{:06d}.jpg'.format(count+1))
        count = count + 1
        
        #count = count + 3
        # create coordinates for each image
        x_min = x - np.round(w/2).astype(int)
        y_min = y - np.round(h/2).astype(int)
        x_max = x + np.round(w/2).astype(int)
        y_max = y + np.round(h/2).astype(int)
        # if image is flipped, coordinates should flip too
        if isflip :            
            x_min = 512 - x_min
            y_min = 512 - y_min
            x_max = 512 - x_max
            y_max = 512 - y_max
            x = 512 - x
            y = 512 - y
        
        # draw coordinate box for lung nodules
        draw = ImageDraw.Draw(im1)
        # bolding the boundaries       
        draw.polygon([(x_min-1,y_min-1),(x_min-1,y_max+1),(x_max+1,y_max+1),(x_max+1,y_min-1)], outline=(255,0,0))
        draw.polygon([(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)], outline=(255,0,0))
        draw.polygon([(x_min+1,y_min+1),(x_min+1,y_max-1),(x_max-1,y_max-1),(x_max-1,y_min+1)], outline=(255,0,0))
        # image names should be like '000001.jpg'
        im1.save('VOCdevkit/VOC2007/LabelImages/{:06d}.jpg'.format(count - 2))
        
        draw = ImageDraw.Draw(im2)       
        draw.polygon([(x_min-1,y_min-1),(x_min-1,y_max+1),(x_max+1,y_max+1),(x_max+1,y_min-1)], outline=(255,0,0))
        draw.polygon([(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)], outline=(255,0,0))
        draw.polygon([(x_min+1,y_min+1),(x_min+1,y_max-1),(x_max-1,y_max-1),(x_max-1,y_min+1)], outline=(255,0,0))
        im2.save('VOCdevkit/VOC2007/LabelImages/{:06d}.jpg'.format(count - 1))

        draw = ImageDraw.Draw(im3)      
        draw.polygon([(x_min-1,y_min-1),(x_min-1,y_max+1),(x_max+1,y_max+1),(x_max+1,y_min-1)], outline=(255,0,0))
        draw.polygon([(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)], outline=(255,0,0))
        draw.polygon([(x_min+1,y_min+1),(x_min+1,y_max-1),(x_max-1,y_max-1),(x_max-1,y_min+1)], outline=(255,0,0))
        im3.save('VOCdevkit/VOC2007/LabelImages/{:06d}.jpg'.format(count))
        
        # save labels as xml format
        xmlLabel = ToXml('{:06d}.jpg'.format(count - 2), x, y, w, h)
        xmlLabel.write('VOCdevkit/VOC2007/Annotations/{:06}.xml'.format(count - 2))
        xmlLabel = ToXml('{:06d}.jpg'.format(count - 1), x, y, w, h)
        xmlLabel.write('VOCdevkit/VOC2007/Annotations/{:06}.xml'.format(count - 1))
        xmlLabel = ToXml('{:06d}.jpg'.format(count), x, y, w, h)
        xmlLabel.write('VOCdevkit/VOC2007/Annotations/{:06}.xml'.format(count))             
        nameLast = namePre
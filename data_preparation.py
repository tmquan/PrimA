import os
import sys
import glob

import argparse
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from natsort import natsorted
from lxml import etree
import cv2
import skimage.io

schema = {
    'Background': 0,
    'TextRegion': 1,
    'ImageRegion': 2,
    'LineDrawingRegion': 3, 
    'GraphicRegion' : 4, 
    'TableRegion' : 5,
    'ChartRegion' : 6, 
    'SeparatorRegion' : 7,
    'MathsRegion' : 8,
    'NoiseRegion' : 9,
    'FrameRegion' : 10,
}

semans = [
    'Background',
    'TextRegion',
    'ImageRegion',
    'LineDrawingRegion', 
    'GraphicRegion', 
    'TableRegion',
    'ChartRegion', 
    'SeparatorRegion',
    'MathsRegion',
    'NoiseRegion',
    'FrameRegion',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    srcFolder = './dataset/'
    dstFolder = './data/'
    dstImageFolder = os.path.join(dstFolder, 'image')
    dstLabelFolder = os.path.join(dstFolder, 'label')

    shutil.rmtree(dstImageFolder, ignore_errors=True)
    shutil.rmtree(dstLabelFolder, ignore_errors=True)
    
    os.makedirs(dstImageFolder, exist_ok=True)
    os.makedirs(dstLabelFolder, exist_ok=True)

    # Take the xml files
    xml_folder = os.path.join(srcFolder, 'XML')
    img_folder = os.path.join(srcFolder, 'Images')

    xml_files = natsorted(glob.glob(xml_folder + '/*.xml'))
    print(xml_files[:1])

    for xml_file in xml_files: #[52:52+1] 718 #[3:3+1] 90 #[9:9+1] 128
        

        tree = etree.parse(xml_file)
        root = tree.getroot()

        imageFilename = None
        imageHeight = None
        imageWidth = None
        # Extract the metadata of image
        for elem in root.getiterator():
            if 'Page' in elem.tag:
                # print(elem.attrib)
                metadata = elem.attrib
                
                imageFilename = str(metadata['imageFilename'])
                imageHeight = int(metadata['imageHeight'])
                imageWidth = int(metadata['imageWidth'])

        
        print(xml_file)
        print(imageFilename)

        if imageFilename and imageWidth>0 and imageHeight>0:
            # Create an empty image
            label = np.zeros([imageHeight, imageWidth])
        
            # Extract the regions
            for elem in root.getiterator():
                for seman in semans:
                    
                    if seman in elem.tag: #if 'TextRegion' in elem.tag:
                        # print(seman)
                        # print(schema[seman])
                        # print(elem.tag)


                        for item in elem.getchildren(): #type
                            # if item is not None:
                            mask = np.zeros(label.shape, dtype=np.bool)
                            vertex_row_coords = []
                            vertex_col_coords = []

                            for coord in item.getchildren():
                                # print(coord.attrib['y'], coord.attrib['x'])
                                vertex_row_coords.append(int(coord.attrib['y']))
                                vertex_col_coords.append(int(coord.attrib['x']))

                            # print(vertex_col_coords)
                            from skimage import draw
                            if vertex_col_coords and vertex_row_coords:
                                fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, 
                                    vertex_col_coords, 
                                    label.shape)
                            
                            mask[fill_row_coords, fill_col_coords] = True
                        offset = 20 # distance between levels: 0~10 -> 0~200
                        label[mask==True] = offset*schema[seman] 
                
         
            # Resize the image
            if os.path.exists(img_folder + '/' + imageFilename):
                image = skimage.io.imread(img_folder + '/' + imageFilename)
                dstImageFilename = dstImageFolder + '/' + imageFilename.replace('tif', 'png')
                dstLabelFilename = dstLabelFolder + '/' + imageFilename.replace('tif', 'png')
                skimage.io.imsave(dstImageFilename, image.astype(np.uint8))
                skimage.io.imsave(dstLabelFilename, label.astype(np.uint8))
 
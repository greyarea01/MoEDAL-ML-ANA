#from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
# I have added mplot imaging // LM
import matplotlib.image as mpimg
import numpy as np
import string
import scipy.ndimage as ndimage
#import matplotlib.patches as patches
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import gaussian_filter

# might not be necessary
#from scipy.ndimage import gaussian_filter
#from scipy.ndimage import convolve
#from scipy.ndimage import convolve1d
#from scipy.ndimage import maximum_position
import os

import argparse
import sys

# Import from image tools directory
# TODO - dependancy removed
#from . import filters

"""
Author - l.millward@qmul.ac.uk
for MoEDAL experiment
"""

class Slide:
    """   
    -----------------------------------------------------------------------------------
    Class for handling Microscope-slide image data, given a canonical file structure of
    'something{foil}something{image number}something{illumination channel}' will read 
    the image data and assemble it into numpy arrays. ID of etch-pits in clean foils
    and basic plotting of coordinates onto foil images may be performed with this class
    -----------------------------------------------------------------------------------
    """
    def __init__(self,foil,number,path = None,flag=None):
        
        if path == None: self.path = './imtest/10c/stack{0:}_s{1:}_{2:}.png'
        else: self.path = path

        self.foil = foil
        self.number = number

        self.b = 0
        self.halo = 0
        self.raw = 0
        self.derived = 0

        # Etch-pit coordinates, if known
        self.x = []
        self.y = []

        """
        Depending on flag, load either backlit image data, or
        full set of rotational image data by default
        """
        self.flag = flag
        if flag=='b': self.loadClean()
        else: self.load()

    """
    Data loading functions
    -----------------------------------------------------
    """

    def fetch(self,step):
        return mpimg.imread(self.path.format(self.foil,self.number,step))

    # Load only backlit images into array (mainly for clean foils)
    def loadClean(self,):
        self.b = self.fetch(9)

    # Load images into array
    def load(self,):
        self.b    = self.fetch(9)
        self.halo = self.fetch(10)
        self.raw  = np.zeros([480,640,8])
        for i in range(0,8): 
            self.raw[:,:,i] = self.fetch(i+1)

    # TODO - Mostly depreciated as of -2021-
    # used for Aligned images only 360 x 300, only channel 1,2 and 9
    def load2(self,):
        self.b    = self.fetch(9)
        #self.halo = self.fetch(10)
        self.raw  = np.zeros([300,360,8])
        for i in range(0,8): 
            self.raw[:,:,i] = self.fetch(i+1)

    """
    Slide processing functions
    -----------------------------------------------------
    """


    def process(self,):
        # Old preprocessing function from Xenon study
        # TODO - move subtract npav to class method
        self.derived = subtract_npav(self.raw)
        

    def getXYclean(self,array=True):
        """
        Find pits in clean foils
        - Note may return Null array if no pits found
        """
        segmask = laplace_filter(self.b)
        XY = coord_list(segmask)
        #self.x, self.y = X,Y
        if array == True: return XY
        else:
            X,Y = XY.T[0],XY.T[1]
        return X,Y 
        

    def loadXY(self,X,Y): self.x,self.y = X,Y

     
    def cropclean(self,X,Y,margin=14):
        # TODO - not necessary, 
        pits = []
        X,Y = remove_margin(X,Y,margin,xmax=360,ymax=300)
        for x,y in zip(X,Y):
            pits.append(np.expand_dims(self.b[x-margin:x+margin,y-margin:y+margin],axis=2))
        return pits
    

    def crop(self,pit,margin=14,flag=None):
        """
        Interacts with 'Etchpit' class, crops at the pit's nominal x,y location
        Remember np.arrays go [y,x]
        """
        a,b = pit.y-margin, pit.y+margin
        c,d = pit.x-margin, pit.x+margin
        
        if flag =='b': 
            pit.b = self.b[a:b,c:d]
        else:
            pit.b = self.b[a:b,c:d]
            pit.h = self.halo[a:b,c:d]
            pit.rim = self.raw[a:b,c:d,...]
            
    def flip(self,):
        """
        Flip images horizontally, to preserve relative orientation in reverse
        foils. Permute the rot_image channels [21876543] to restore symmetry 
        """
        self.b = np.flip(self.b,axis=1)
        if self.flag == None:
            self.halo = np.flip(self.halo,axis=1)
            self.raw = np.flip(self.raw,axis=1)
            #// permute the rot_image channels - 21876543
            self.raw = np.roll(self.raw,shift=-2,axis=2)
            self.raw = np.flip(self.raw,axis=2)  

    def renormalise(self,av_image,flag=None):
        """
        If we have a standardised/average image for the dataset we can renormalise
        the image zero relative to this to remove the systematic bias.
        """
        self.b -= av_image.b
        if self.flag == None:
            self.halo -= av_image.halo
            self.raw -= av_image.raw
            
    def r_avg(self,):
        """
        Average of the rotational imaging channels
        """
        self.r_avg = np.mean(self.raw,axis=-1)
        #print(np.shape(self.r_avg))
        return self.r_avg
        

"""
 Simple 3D image class - used for handling dataset averages
"""
class Multi_Channel_Image:
    def __init__(self,image):
        if len(image.shape) ==2:
            self.b = image
        else:
            self.b = image[...,8]
            self.halo = image[...,9]
            self.raw = image[...,0:8]
            

"""
========================================================
    Old image tool functions

========================================================
"""

def plot_coords(X,Y,img,icon = 'ro',show=True,title=None,save=None):
    """
    Plot x,y coordinates and background image
    Disabling show will display all plots at the end
    """
    fig, ax = plt.subplots()
    ax.imshow(img,cmap='gray')
    #TODO
    # Was Y,X for consistency w earlier work. changed 11 mar 2021
    ax.plot(X,Y,icon)
    if title != None: plt.title(title)
    if save != None: plt.savefig(save)
    if show == True: plt.show()

def plot(img,show=True,title=None,cmap=None,save=None):
    fig, ax = plt.subplots()
    if cmap == None: ax.imshow(img)#,cmap='gray')
    else: ax.imshow(img,cmap)
    if title != None: plt.title(title)
    if save != None: plt.savefig(save)
    if show == True: plt.show()

def save_coords(X,Y,img,directory,gray=True,box=False):
    """
    LM - Added Greyscale default
    """
    fig, ax = plt.subplots()
    if gray == True:  ax.imshow(img,cmap = 'gray')
    if gray == False: ax.imshow(img)
    if box != False: ax.add_patch(box)
    ax.plot(Y,X,'ro')
    plt.savefig(directory)
    plt.close()


"""
--------------------------------------------------------------------
	Not sure where used: A: used in process
--------------------------------------------------------------------
"""

def subtract_npav(npimg):
    npav = average_npimg(npimg)
    npimg2 = np.copy(npimg)
    for i in range(npimg.shape[-1]):
        npimg2[...,i] = (npimg[...,i]-npav)
    return npimg2 

def average_npimg(npimg):
  return np.mean(npimg,axis=-1) # last axis is channels
  #return sum(img)*0.125


"""
--------------------------------------------------------------------
	Filters
--------------------------------------------------------------------
"""

def laplace_filter(im,alpha=0.001,beta=0.00001,sigma=8):
    """
    basic filter to identify pits in clean foil, defined in dataset tools 
    / filters, but redefined here for standalone use.
    Double application of gauss-laplace and filter / threshold ID's pits
    """
    z = gaussian_laplace(im,sigma = sigma)
    z = z*(z>alpha)  # 0.001 is about the order of magnitude suitable here
    z = gaussian_laplace(z,sigma = sigma)
    z = (z < -beta)
    return z


def coord_list(mask):
    """
    For an image segmentation mask it generates a coordinate for the 
    centre of mass of each detatched region 
    
    Returns (N,2) array of X,Y points
    if there are no regions it returns []. Use array.size == 0 to test if this is empty or not
    """
    labs, nlab = ndimage.label(mask)
    XY = ndimage.center_of_mass(mask,labs,range(1,nlab+1)) 
    XY = np.flip(XY,axis=-1).astype(int)
    return XY
        

def remove_margin(X,Y,margin,xmax=640,ymax=480):
    """
    Note! - rarely used, X and Y may be reversed
    """
    xnew,ynew = [],[]
    for i in range(len(X)):
        # test its not out of bounds
        if (X[i]-margin) < 0 or (X[i]+margin) > xmax :
            continue
        if (Y[i]-margin) < 0 or (Y[i]+margin) > ymax :
            continue
        xnew.append(X[i])
        ynew.append(Y[i])
    return xnew,ynew



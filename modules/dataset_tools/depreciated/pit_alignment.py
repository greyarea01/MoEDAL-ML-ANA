
#from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
# I have added mplot imaging // LM
import matplotlib.image as mpimg
import numpy as np
import string
import scipy.ndimage as ndimage
# might not be necessary
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from scipy.ndimage import convolve1d
from scipy.ndimage import maximum_position
import os

import argparse
import sys

from . import filters
from . import mhio
from . import plotting
from . import image_tools as it




class Offset:
    """
    -------------------------------------------------------------
    Corrects foil alignment offset between different foils

    Takes X,Y pit coords from one foil-scan and converts to 
    the X,Y coordinate of that pit in other foil-scans

    Misalignment comes from;
    a) mechanical misalignment of foils during scanning
       - Fixed displacement 
       + rotational element that scales with global X,Y
    b) ion track angle of incidence, (fixed for all test beam tracks, 
       and fixed relative to origin for signal tracks in production foils)

    -------------------------------------------------------------
    # IDEA - might be easier to flip the reverse images entirely
    # NO! - will not be possible to compare multichannel images
    """
    #FIXME - X,Y are mislabelled

    def __init__(self,preset):
        # Default offset
        self.Xscale = 11
        self.Yscale = 14
        self.Xzero = 0
        self.Yzero = 0
        self.invertX = False
        # Image/scan parameters
        self.line = 25 # how many i steps in each scan line
        self.width = 640
        self.height = 480
        self.Xoverlap = 15 # how many pixels do 2 images overlap 
        self.Yoverlap = 15

        """
        Preset coordinate conversions
        cd = clean foil, to dirty/exposed
        cdr = clean foil, to dirty/exposed reverse
        """
        if preset == 'cd': self.reset(-10,13,7,-7)
        if preset == 'cdr': self.reset(-0.4,-0.9,9,9,invertX=True)

        if preset == 'dc': self.reset(10,-13,-7,7)
        

    def Xoff(self,i): return int(self.Xzero +  int(i/self.line)*self.Yscale)

    def Yoff(self,i): return int(self.Yzero + (i%self.line)*self.Xscale)

    def convert_ixy(self,z):
        """
        Convert X,Y coordinates from one scan to another
        """
        inew,Xnew,Ynew = z[0],z[1],z[2]
        Xnew = z[1] + self.Xoff(z[0])
        if self.invertX == True:
            Ynew = (self.width-z[2] + self.Yoff(z[0]))
        else:
            Ynew = z[2] + self.Yoff(z[0])

        """
        Incrementing slide number 
        -------------------------
        ! No protection built in for offsets > 1 slide v/h !
        ! No protection for negative offsets on initial slide !
        EDGECASE - i becomes negative
        """
        if Ynew > self.width:
            inew += self.line
            Ynew = Ynew%self.width  -self.Yoverlap
        if Ynew < 0:
            inew -= self.line
            Ynew = Ynew + self.width - self.Yoverlap

        if Xnew > self.height:
            inew +=1
            Xnew = Xnew%self.height + self.Xoverlap
        if Xnew < 0:
            inew -=1
            Xnew = self.height+Xnew +self.Xoverlap

        return (inew,Xnew,Ynew)

    def reset(self,Xscale,Yscale,Xzero,Yzero,invertX=False):
        self.Xscale = Xscale
        self.Yscale = Yscale
        self.Xzero = Xzero
        self.Yzero = Yzero
        self.invertX = invertX



def overlap(z1,z2,tolerance=16.):
    if abs(z1[1]-z2[1])<tolerance and abs(z1[2]-z2[2])<tolerance: 
        return True
    else:
        return False

def confirm(list1,list2,tolerance=16.):
    return [z1 for z1 in list1 for z2 in list2 if (overlap(z1,z2)==True)]



def clean_locations(
        start = 2,
        stop = 251,
        exclude = [195,194,351,376],
        path = '/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png',
        outpath = 'pit_locations.txt',
    ):
    """
    Functions
    """
    def fetch_pits(foil,i):
        clean = it.Slide(foil,i,path)
        clean.loadClean()
        X,Y = clean.getXYclean()
        return [(i,x,y) for x,y in zip(X,Y)]  


    """
    Clean location script
    1) Identifies etch-pit locations in clean foils (using image tools)
    2) Compares locations between foil 1 front and reverse
    """

        # Output etch-pit locations
    pits = { 
        'f1hits' : 0, 'f1rhits' : 0, 'f2hits' : 0,
        'Confirmed pits' : [],
        'Double Confirmed pits' : [],
        'Single foil pits' : [],
        'Single side pits' : [],
        }

    """
    slide exclusion
    these slides correspond to the two alignment holes (beside the initial)
    """
    exclude = [195,194,351,376]


    for slide in range(start,stop): #251): # skip alignment hole image
        if slide in exclude:
            continue
         
        """
        Get clean foil locations
        !NOTE! 
        clean foils do not have an alignment correction
        however natural alignment is good enough that they are ok to within tolerance
        """
        foil1 = fetch_pits('clean1_s',slide)
        foil1r = list(map(lambda z: (z[0],z[1],640-z[2]),fetch_pits('clean1_r_s',slide)))
        foil2 = fetch_pits('clean2_s',slide)

        pits['f1hits']  += len(foil1)
        pits['f1rhits'] += len(foil1r)
        pits['f2hits']  += len(foil2)

        confirmed = confirm(foil1,foil1r)
        double_confirmed = confirm(confirmed,foil2)
        single_foil = [pit for pit in confirmed if pit not in double_confirmed]
        single_side = [pit for pit in foil1 if pit not in confirmed]

        pits['Confirmed pits'].extend(confirmed)
        pits['Double Confirmed pits'].extend(double_confirmed)
        pits['Single foil pits'].extend(single_foil)
        pits['Single side pits'].extend(single_side)

    """
    tally final count
    """
    keys = ['Confirmed pits','Double Confirmed pits','Single foil pits','Single side pits']
    for key in keys: 
        print(len(pits[key]),'	'+key) 

    """
    Export locations as Json
    """
    with open(outpath,'w') as outfile:
        json.dump(pits,outfile)

    return pits


def convert_locations():
    # 22jan 2021
    # converts clean locations to exposed locations
    off = Offset('cd')
    keys = ['Confirmed pits','Double Confirmed pits','Single foil pits','Single side pits']

    pit_locations = mhio.load_json('./pit_locations.txt')

    for key in keys: 
        exposed_locations = list(map(off.convert_ixy,pit_locations[key]))
        pit_locations[key] = exposed_locations

    mhio.save_json(pit_locations,'pit_locations_exposed.txt')



def alignment_mini_test(): 
    pit_locations = mhio.load_json('./pit_locations_exposed.txt')['Confirmed pits']

    # fetch 251 pits
    pits = [ z for z in pit_locations if z[0]==134]
    i,x,y = zip(*pits)
    
    path = '/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png'
    clean = it.Slide('dirty_s',134,path)
    #clean = it.Slide('clean1_s',291,path) 
    clean.loadClean()
    it.plot_coords(x,y,clean.b,title='Original',show=True)


def offset_per_slide():
    """
    Print out all slide no.s and corresponding offset
    """
    off = Offset('cd')
    for i in range(1,251):
        print('{} : x {}	y {}'.format(i,off.Xoff(i),off.Yoff(i)))
        # Max available aligned image area
        #print('{} : x {}	y {}'.format(i,480-off.Xoff(i),640+off.Yoff(i)))

def reverse_test():
    """
    plot a transform, and reverse it
    """
    z = (172,45,67)
    print(z)
    off = Offset('cd')
    print(off.Xoff(z[0]),off.Yoff(z[0]))
    znew = off.convert_ixy(z)
    print(znew)
    #off.reset(0,0,-59,237)
    off.reset(10,-13,-7,7)
    z2 = off.convert_ixy(znew)
    print(z2)


def alignment_test():
    """
    Alignment correction unit test
    ----------------------------------------------------------------------
    1) Take a few pits, plot their uncorrected alignment
    2) Apply alignment correction for exposed foil, & plot
    3) Apply alignment correction for reverse-side of exposed foil, & plot
    4) Plot the coordinates before / after correction
    """

    """
    Get X,Y test coords
    """
    path = '/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png'
    s = 296

    def original_coords():
        clean = it.Slide('clean1_s',s,path) 
        clean.loadClean()
        x,y = clean.getXYclean()
        it.plot_coords(x,y,clean.b,title='Original',show=False)
        print('old x',x)
        print('old y',y)
        return [(s,X,Y) for X,Y in zip(x,y)]

    def correction(Z):
        """
        Test clean to dirty, alignment correction
        """
        dirty = it.Slide('dirty_s',s,path)
        dirty.load()
        #it.plot_coords(x,y,dirty.b,title='uncorrected coordinates',show=False)

        off = Offset('cd')
        #off.clean_to_dirty()
        Z2 = map(off.convert_ixy,Z)
        i,x,y = zip(*Z2)
        print('new i',i)
        print('new x',x)
        print('new y',y)

        it.plot_coords(x,y,dirty.b,title='corrected coordinates',show=False) 
        dirty2 = it.Slide('dirty_s',s,path)
        dirty2.load()
        it.plot_coords(x,y,dirty2.b) 

        # Test inverse correction - ie, going dirty to clean
        #off2 = Offset('dc')
        #Z3 = map(off2.convert_ixy,Z2)
        #i,x,y = zip(*Z3)
        #print('old i',i)
        #print('old x',x)
        #print('old y',y)
        #it.plot_coords(x,y,dirty2.b) 
      

    def reverse_correction(Z):
        flipped = it.Slide('dirty_r_s',s,path) 
        flipped.loadClean()
        cr = it.Slide('clean1_r_s',s,path) 
        cr.loadClean()
        """
        Test reverse image, alignment correction
        """
        off = Offset('cdr')
        Z3 = map(off.convert_ixy,Z)
        i,x,y = zip(*Z3)
        print('new i',i)
        print('new x',x)
        print('new y',y)
        it.plot_coords(x,y,flipped.b,show=False,title='Flipped coordinates + offset')

    Z = original_coords()
    correction(Z)
    #reverse_correction(Z)
    plt.show()
    quit()















import matplotlib.image as mpimg
import numpy as np
import string
import os



import argparse
import sys

from . import filters
from . import mhio
from . import plotting
from . import image_tools as it
from . import pit_alignment


class EtchPit:
    """
    Class to handle etch-pits and associated data
    """

    def __init__(self,z):
        self.i = z[0]
        self.x = z[1]
        self.y = z[2]

        # Etch pit image data
        self.b = 0
        self.h = 0
        self.rim = 0

        self.clean_location = z
        self.exposed_location = None
        self.reverse_location = None

    def z(self,):
        # gives location as tuple-coordinate
        return (self.i,self.x,self.y)

    def set_z(self,z):
        self.i = z[0]
        self.x = z[1]
        self.y = z[2]

    #def plot(self,):
    ## Undecided if merits own plt function




  


class Production:
    """
    Class to handle pre-production of ML datasets from raw etch-pit data
    Typically this will take png scan data (converted by file-converter)
    It will crop this data around XY etch-pit locations, and output a set of cropped etch-pits 

    ['Confirmed pits','Double Confirmed pits','Single foil pits','Single side pits']
    """
    def __init__(self,):

        # / path to png files / { foil prefix }{ slide position }_{ image channel }
        self.path = '/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png'
        self.outpath = '/home/millward/MLnov20/pits/feb_pits/{}'
        self.slides = range(2,251)
        # TODO - handle alignment hole exclusion as a coordinate exclusion around the holes
        self.exclude = [195,194] # slides to exclude due to containing an alignment hole
                                 # [376,351]

        # Image and pit shape parameters
        self.pit_shape = (28,28,8)
        self.margin = 14
        self.width = 640
        self.height = 480


    def margin_and_positive(self,z):
        """
        test if a pit is within correct image margin
        """
        # remove negative values
        if z[1] < 0 or z[2] < 0 : return False
        # remove values out of margin
        if z[1] < self.margin or z[2] < self.margin : return False
        if z[1] > (self.height-self.margin) : return False
        if z[2] > (self.width-self.margin) : return False
        return True

    #def convert_location(self,)

    def get_signal(self,location_file='./pit_locations.txt',key='Confirmed pits',offset='cd'):
        """
        1) Load pit locations: tuples (i,x,y)
        2) Correct for alignment offset, & check margin
        3) Convert from tuples to 'EtchPit' class, and return
        """
        pit_locations = mhio.load_json(location_file)[key]
        #etchpits = [EtchPit(pit) for pit in pit_locations]
        # etch-pit has correct 

        off = pit_alignment.Offset(offset)
        exposed_locations = list(map(off.convert_ixy,pit_locations))
        print('sig pits :',len(exposed_locations))
        passed = list(filter(self.margin_and_positive,exposed_locations))
        print('sig pits in margin:',len(passed))

        etchpits = [EtchPit(pit) for pit in passed]
        return etchpits


    def get_background(self,slides=[200,197,191,190,173,169,219]):
        """
        1) Take set of slides known to contain no signal
        2) Take sex of X,Y grid points from those slides
        
        """
        emptyslides = slides
        #[200,197,196,191,190,173,169] #,219,220,224,135,125,122,119]

        fake_pits = []
        for empty in emptyslides:
            X,Y = XYgrid()
            z = [ (empty,x,y) for x,y in zip(X,Y) ]
            fake_pits.extend(z)
        print('No. bkg Examples:',len(fake_pits))

        etchpits = [EtchPit(pit) for pit in fake_pits]
        return etchpits


    def output_pits(self,out_pits,name,outpath=None):
        print(name+' :',len(out_pits))
        if outpath == None: outpath = self.outpath
        mhio.pickle_dump(out_pits,outpath.format(name))


    def slide_loop(self,i,pits,foil='dirty_s'):
        """
        Loads the slide data, and crops corresponding pits.
        This data is added to the pit class
        """
        if i%10==0: print('Processing slide ',i)
        s = it.Slide(foil,i,self.path)
        s.load()

        ## Commented out code previews which pits will be cropped
        #X,Y = zip(*[(p.x,p.y) for p in pits if p.i == i])
        #it.plot_coords(X,Y,s.b,title='Preview')

        for p in pits:
            if p.i==i: s.crop(p) 
        del s


    def run_production(self,):
        """
        Run default production job
        """
        sig = self.get_signal('./pit_locations.txt','Double Confirmed pits')
        bkg = self.get_background()

        for slide in self.slides:
            self.slide_loop(slide,sig)
            self.slide_loop(slide,bkg)

        crop_ok = (lambda x : np.shape(x.rim)==(28,28,8))

        sig_passed = list(filter(crop_ok,sig))
        print(len(sig_passed), 'len sig out')
        self.output_pits(sig_passed,'sig_pits')

        bkg_passed = list(filter(crop_ok,bkg))
        print(len(bkg_passed), 'len bkg out')
        self.output_pits(bkg_passed,'bkg_pits')


"""
 Derived production class
"""
class ProductionReverse(Production):

    """
    Take existing - cropped pits
    Re-run the conversion / alignment for the reverse side and add new data
    """

    def Add_reverse_data2(self,):

        sig_orig = mhio.pickle_load(pit_file)
        bkg_orig = mhio.pickle_load(pit_file)

        for slide in self.slides:
            self.slide_loop(slide,sig)
            self.slide_loop(slide,bkg)

        crop_ok = (lambda x : np.shape(x.rim)==(28,28,8))

        sig_passed = list(filter(crop_ok,sig))
        print(len(sig_passed), 'len sig out')
        self.output_pits(sig_passed,'sig_pits')

        bkg_passed = list(filter(crop_ok,bkg))
        print(len(bkg_passed), 'len bkg out')
        self.output_pits(bkg_passed,'bkg_pits')



    def Add_reverse_data(self,):
        """
        Load Nominal clean locations
        """
        sig_orig = mhio.load_json('./pit_locations.txt')['Double Confirmed pits']
        sig_orig = [EtchPit(pit) for pit in sig_orig]
        bkg_orig = self.get_background()

        """
        Convert these locations to exposed and reverse exposed
        """

        def convert(pits):
            front_off = pit_alignment.Offset('cd')
            back_off = pit_alignment.Offset('cdr')

            for x in pits:
                x.exposed_location = front_off.convert_ixy(x.clean_location)
                x.reverse_location = back_off.convert_ixy(x.clean_location)

        convert(sig_orig)
        convert(bkg_orig)

        print('conversion done')
        print(sig_orig[0].exposed_location)
        print(sig_orig[0].reverse_location)

        # copy pit
        pitA = EtchPit(sig_orig[0])


        align = (lambda x : self.margin_and_positive(x.exposed_location) and
                            self.margin_and_positive(x.reverse_location))

        sig_ok = list(filter(align,sig_orig))
        bkg_ok = list(filter(align,bkg_orig))




        quit()


        
        #def split():



        pit.set_z()

        off = pit_alignment.Offset(offset)
        exposed_locations = list(map(off.convert_ixy,pit_locations))
        print('sig pits :',len(exposed_locations))
        passed = list(filter(self.margin_and_positive,exposed_locations))
        print('sig pits in margin:',len(passed))

        etchpits = [EtchPit(pit) for pit in passed]
        return etchpits








"""
production utilities
"""
def make_images(pit_file='./pits/feb_pits/sig_pits'):
    """
    Make images from 
    """
    pits = mhio.pickle_load(pit_file)

    for p in pits:
        it.plot(p.b,cmap='gray',save='./pits/test_s/pb',show=False)

def validate(pit_file='./pits/feb_pits/sig_pits_1306'):
    pits = mhio.pickle_load(pit_file)
    for p in pits:
        if np.shape(p.rim)!=(28,28,8): print('error r',np.shape(p.rim),p.i)
        if np.shape(p.h)!=(28,28): print('error h',np.shape(p.h),p.i)
        if np.shape(p.b)!=(28,28): print('error b',np.shape(p.b),p.i)
    


def XYgrid():
    """
    can use larger grid dimensions
    TODO - add smarter options
    """
    X = []
    Y = []
    for j in range(0,15):
        for k in range(0,15):
            x = 30*(j+1)
            y = 30*(k+1)
            X.append(x)
            Y.append(y)
    return X,Y

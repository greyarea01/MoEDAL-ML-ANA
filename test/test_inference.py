from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from dataset_tools import image_tools as it
from dataset_tools import mhio
import inference_tools as inf 


from scipy.ndimage import gaussian_laplace
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
import scipy.ndimage as ndimage

"""
Author - l.millward@qmul.ac.uk
for MoEDAL experiment
"""

def test_inf():
    """
    1) Create ensemble of models
    """
    FCNN = inf.Ensemble('./MLout/fold-{0}.h5',9,(480,640,8))
    slide_path = '/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png'
    foil = 'dirty_s'
    heatmap_output = {}

    """
    2) Generate inference predictions for each microscope-slide
    """
    for i in range(2,99):
        slide = it.Slide('dirty_s',i,slide_path)
        heatmap_output[i] = FCNN.predict_slide(slide.raw)
        #it.plot(heatmap_output[i][:,:,0])  # Optional -to plot a heatmap
        del slide

    """
    3) Output is a dictionary, key i = slide / image number
       each entry is a (h,w,c) array with each channel c representing the heatmap for 
       the c'th model in the ensemble - ie (60x80x5)
    """
    mhio.pickle_dump(heatmap_output,'./MLout/heatmap_outputs')


def scale(x,y,ratio=80./640.):
    """
    Take a list of x,y coordinates and rescale
    used to change between 60x80 heatmaps and 480x640 images
    """
    return [ X*ratio for X in x], [ Y*ratio for Y in y]


def process_heatmap(Z):
    """
    Take (60,80,5) heatmap array for i'th slide and produce a set of predicted x,y etch-pit locations.
    The locations predicted depend on the method used for deriving x,y coordinates
    """

    """
    Calc. average image, and gaussian laplace (of each channel) by default
    """
    zav = np.average(Z,axis=2)
    gauss = [gaussian_laplace(Z[:,:,j],sigma = 1.0) for j in range(0,5)]
    #it.plot(zav)

    #x,y = it.coord_list(im>th)
    #it.plot_coords(x,y,(im>th))

    def default_opt(Z,thresh=0.1):
        """
        Average of gaussain laplace functions, w. threshold of 0.1
        """
        gs = sum(gauss) # average the gaussians
        x,y = it.coord_list(-gs>thresh)
        #it.plot(-gs>0.1)
        return x,y

    def opt2(Z,thresh=0.1):
        """
        No ensembling / average, just gaussian laplace for first model in ensemble
        """
        it.plot(-gauss[0]>0.1)
        return it.coord_list(-gauss[0]>0.1)

    #X,Y = default_opt(Z)
    X,Y = opt2(Z)
    X,Y = scale(X,Y,640./80.)
    return X,Y


def predict_locations(path='./MLout/heatmap_outputs',fixedi=None):
    """
    Load predicted heatmaps, key corresponds to predictions for a slide 'i'
    """
    hmaps = mhio.pickle_load('./MLout/heatmap_outputs')
    for i in hmaps.keys():
        x,y = process_heatmap(hmaps[i])
        s = it.Slide('dirty_s',i,'/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png','b')
        it.plot_coords(x,y,s.b)
        #it.plot_coords(x,y,s.b,save='save images here')


"""
Run stand alone, or import into other script
"""
#test_inf()
#predict_locations()


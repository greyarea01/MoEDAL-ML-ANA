
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.ndimage import gaussian_filter
#from modules.dataset_tools import image_tools as it
#from modules.dataset_tools import mhio

import image_tools as it
import mhio


#class options:
#    def __init__(self):
#        self.foil
#        self.path

options = {




}
"""
Need to generate image averages for the following foil-scans.

 Backlit channel only
 --------------------------------------------
 clean, clean-2, clean-r
 clean-r flipped can be produced by flipping

 Backlit channel, Halo, and rotational channels
 --------------------------------------------
 dirty, dirty-r
 dirty-r flipped, produced via flipping

"""


class Average_image:

    path = '/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png'
    exclude = [195,194,351,376]
    
    def __init__(self,foil,flag=None):
        """
        Want to remove central illumination bias from dataset.
        Add together all images in dataset and smooth with gauss (sig >> pit).
        """
        self.foil = foil #//
        self.flag = flag #// 
        self.sn = 0
        self.b = 0
        self.halo = 0
        self.raw =0
        
        """
        # Note: I tested difference between average of 250 vs 500 : negligible
        # Just be consistent about which is used.
        """ 
        for i in range(2,250):
        
            if i in self.exclude: continue
            self.accumulate(i)
        self.normalise()
                    
    def accumulate(self,i):
        if self.flag == 'b':
            self.b += it.Slide(self.foil,i,self.path,'b').b
        else:
            s = it.Slide(self.foil,i,self.path)
            self.b += s.b
            self.halo += s.halo
            self.raw += s.raw
        self.sn += 1
              
    def normalise(self,):
        self.b = self.b/(self.sn*1.)
        if self.flag == None:
            self.halo = self.halo/(self.sn*1.)
            self.raw = self.raw/(self.sn*1.)
            
    def flip(self,):
        self.b = np.flip(self.b,axis=1)
        if self.flag == None:
            self.halo = np.flip(self.halo,axis=1)
            self.raw = np.flip(self.raw,axis=1)
            
    def smooth(self,):
        self.b = gaussian_filter(self.b,sigma=28)
        if self.flag == None:
            self.halo = gaussian_filter(self.halo,sigma=50)
            for c in range(0,8):
                self.raw[...,c] = gaussian_filter(self.raw[...,c],sigma=50)


def generate_averages():
    """
    One-time script to generate unsmoothed average images from datasets
    This can be loaded to plot and investigate. Smoothed or flipped images
    can be obtained later.
    """

    Avg = {}

    Avg['c'] = Average_image('clean1_s','b')
    Avg['c2'] = Average_image('clean2_s','b')
    Avg['cr'] = Average_image('clean1_r_s','b')
    Avg['d'] = Average_image('dirty_s')
    Avg['dr'] = Average_image('dirty_r_s')
    print('generated averages')
    
    mhio.pickle_dump(Avg,'../average_images/Avg')


def investigate_averages():
    Avg = mhio.pickle_load('../average_images/Avg')

    def plot_rot(img):
        it.plot(img.halo,title='Avg image (Halo) {}'.format(img.foil))
        for ch in range(0,8):
            it.plot(img.raw[...,ch],title='Avg image (R:{0}) {1}'.format(ch,img.foil))

    for img in Avg.values():
        img.smooth()

    for img in Avg.values():
        it.plot(img.b,title='Avg image (Blit) {}'.format(img.foil))
        if img.flag == None: plot_rot(img)
            

        
    """
    Consider Intensity Spectra
    """
    
    B = Avg['c'].b
    B = B.flatten()
    plt.hist(B,bins='auto')
    plt.show()
    
    Smoothed_Avg = {}
    for key in ['c','c2','cr']: Smoothed_Avg[key] = Avg[key].b
    for key in ['d','dr']:
        Smoothed_Avg[key] = np.zeros([480,640,10])
        Smoothed_Avg[key][...,0:8] = Avg[key].raw
        Smoothed_Avg[key][...,8] = Avg[key].b
        Smoothed_Avg[key][...,9] = Avg[key].halo
    
        
    #mhio.pickle_dump(Smoothed_Avg,'../average_images/Smoothed_Avg')

#investigate_averages()





    












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







def sept_production_aman():
    """
    21 sept, stack some files together as a dataset for aman
    """




    c = []# clean
    d = []# dirty
    p = []# processed

    off = pit_alignment.Offset('cd')

    for i in range(41,42): # skip alignment hole image
        if i == 195: # another alignment hole
            continue

        x = off.Xoff(i)
        y = off.Yoff(i)

        
        print('{}	{}	{}'.format(i,x,y))

        print('Processing slide ',i)

        clean = it.Slide('clean1',i,'/home/millward/Moedal_data/febdat/png_nov/{0:}_s{1:}_{2:}.png')
        clean.loadClean()

        dirty = it.Slide('dirty',i,'/home/millward/Moedal_data/febdat/png_nov/{0:}_s{1:}_{2:}.png')
        dirty.load()

        W = 640
        H = 480
        #x = 100  # is this x or y ( this is y/h dim)
        #y = 0

        # s41 

        def crop_x(img,x,y): #,W,H,x,y):
            img = img[x:H+x,y:W+y,:] 
            it.plot(img[:,:,0])
            return img

        output = np.zeros([480,640,1])
        output[:,:,0] = clean.b
        crop_x(output,0,-y)

        output = np.zeros([480,640,1])
        output[:,:,0] = dirty.b
        crop_x(output,x,0)


        quit()


        output[:,:,8] = dirty.b


        clean2 = it.Slide2('cc',i,'/home/millward/Moedal_data/febdat/aligned/{0:}{1:}_{2:}.png')
        clean2.loadClean()

        outputC = np.zeros([300,360,2])
        outputC[:,:,0] = clean.b
        outputC[:,:,1] = clean2.b
        c.append(outputC)

        dirty = it.Slide2('d',i,'/home/millward/Moedal_data/febdat/aligned/{0:}{1:}_{2:}.png')
        dirty.load2()
        dirty.process() # If want averaged image

        # OUTPUT

        output = np.zeros([300,360,10])
        output[:,:,0:8] = dirty.raw
        output[:,:,8] = dirty.b
        output[:,:,9] = dirty.halo 
        print(np.shape(output))

        d.append(output)

        output2 = np.zeros([300,360,10])
        output2[:,:,0:8] = dirty.derived
        output2[:,:,8] = dirty.b
        output2[:,:,9] = dirty.halo 
        print(np.shape(output2))

        p.append(output2)


    d2 = np.asarray(d)
    print(np.shape(c))
    print(np.shape(p))
    print(np.shape(d))
    print(np.shape(d2))


    pickle_dump(c,'/home/millward/Moedal_data/febdat/aman/stacked_clean_foils')
    pickle_dump(p,'/home/millward/Moedal_data/febdat/aman/exposed_foil_processed')
    pickle_dump(d,'/home/millward/Moedal_data/febdat/aman/exposed_foil')

    print('Done')
    quit()

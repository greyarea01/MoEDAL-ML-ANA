import numpy as np
import scipy.ndimage as ndimage

from scipy.ndimage import convolve
from scipy.ndimage import convolve1d
from scipy.ndimage import maximum_position

# for plotting, shouldnt really be here
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Specialised filters
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import generic_laplace # overcomplex
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel

from scipy.ndimage import maximum

from scipy.ndimage import gaussian_filter1d


from modules.dataset_tools import image_tools as it
from modules.dataset_tools import mhio

"""
	Filters -2 - attempt at producing new / more effective image filters
"""

def test_filters():
    # Load test slides
    s = it.Slide('dirty_s',31,'/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png')
    s2 = it.Slide('clean1_s',31,'/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png','b')
    it.plot(s2.b,title='eg, Image #31',save='./scratch/s31_greenscale.png')
    av_im = generate_average(foil='clean1_s',
                             path='/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png',
                             flag='b')
    new = s2.b-av_im
    it.plot(new,title='#31 corrected',save='./scratch/s31_corr_g.png')
    it.plot(new,cmap='gray',title='#31 corrected',save='./scratch/s31_corr.png')
    sc = new

    """
    Repeat for backlit exposed data
    """
    av_im = generate_average(foil='dirty_s',
                             path='/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png',
                             flag='b')
    it.plot(s.b,title='eg, Image #31',save='./scratch/s31_greenscale.png')
    new = s.b-av_im
    it.plot(new,title='#31 corrected',save='./scratch/s31_corr_g.png')
    it.plot(new,cmap='gray',title='#31 corrected',save='./scratch/s31_corr.png')

    # Pickle the average and corrected images
    mhio.pickle_dump((s2.b,sc),'testslides')


def generate_average(foil,path,flag,plot=False):
    """
    Want to remove central illumination bias from dataset
    Add together all images in dataset and smooth with gauss (sig >> pit)
    """

    av_im = 0
    exclude = [195,194,351,376]
    sn = 0
    for i in range(2,250):
        if i in exclude: continue
        s = it.Slide(foil,i,path,flag)
        av_im += s.b
        sn += 1
    av_im = av_im/(sn*1.)
    if plot:
        it.plot(av_im,title='Backlit dataset - Average image',save='./scratch/ds_average_image.png')
    """
    Smooth the average image
    """
    av_im = gaussian_filter(av_im,sigma=50)
    if plot:
        it.plot(av_im)
    return av_im
    

def test_filters2():
    # load test slide and image
    #s = mhio.pickle_load('testslides')
    #sb,sc = s[0],s[1]

    #it.plot(laplace_filter(sb))
    #it.plot(laplace_filter(sc))

    av_im = generate_average(foil='clean1_s',
                             path='/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png',
                             flag='b')

    """
    Cycle through clean images and test their pit ID
    """
    exclude = [195,194,351,376]
    for i in range(2,250):
        if i in exclude: continue
        s = it.Slide('clean1_s',i,'/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png','b')
        #it.plot(s.b)
        #it.plot(laplace_filter(s.b))  
        #it.plot(laplace_filter(s.b-av_im))  
        sc = s.b-av_im
        z = gaussian_laplace(sc,sigma = 4)
        z2 = gaussian_laplace(sc,sigma = 2)
        z3 = gaussian_laplace(s.b,sigma = 2)
        it.plot(z)
        it.plot(sc)
        zc = z*sc
        it.plot(zc)  #) This is a relatively sharp highlighting of pits
        it.plot(gaussian_filter(zc,sigma = 4))
        print(maximum(sc))
        sobx = sobel(sc,axis=0)
        soby = sobel(sc,axis=1)
        hyp = np.hypot(sobx,soby)
        it.plot(gaussian_filter(hyp,sigma = 4))
        it.plot(hyp*zc)
        #it.plot(z3)

def gf1d(x):
    """
    Gaussian smear in channels
    """
    return gaussian_filter1d(x,sigma=4,axis=-1,mode='wrap')

def laplace_filter(im):
    z = gaussian_laplace(im,sigma = 8)
    # Gaussian laplace is useful for identifying pits in clean foil
    z = z*(z>0.001)  # 0.001 is about the order of magnitude suitable here
    # This is suitable for pit finding, may want to reaply filter
    z = gaussian_laplace(z,sigma = 8)
    """
    Double application of laplace and filter / threshold ID's pits
    """
    z = (z < -0.00001)
    # This is the segmentation mask (for clean)
    return z



def segM3(raw,npcube,back,halo):
    """
    Try other tools eg conv, from scipy ndimage
    """
    #img = img_convert_npy(npcube)
    #z = [ gaussian_filter(img[i],sigma=2) for i in img ]
    z = gaussian_filter(raw,sigma=2)
    z2 = gaussian_filter(npcube,sigma=2)
    #av = sum(z)
    xy = maximum_position(z)
    print(xy)
    return z[:,:,0],xy



def my_convolve1d(im):
    """
    test function for ndimage conv
    """
    kernel = np.array( (1.,0.,0.,0.,1.) )
    x = convolve1d(im,kernel)
    y = convolve1d(im,kernel)
    #z = gaussian_laplace(im,sigma = 3)
    #plt.imshow(x)
    #plt.show()
    #plt.imshow(y)
    #plt.show()
    plt.imshow(z)
    plt.show()
    return z
    # print out the respective convoloution

"""
  Suggestion:
  should implement as class based filters, with option for returning as coordinates or as mask

"""



"""
 Scipy Ndimage toolys;
convolve
- specify sprial kernel
convolve1d
gaussian

maximum
maximum position
mean 
median
minimum
minimum position

gaussian laplace
generic laplace
sobel
gaussian gradient magnitude

"""


#-----------------------------------------------
# Filters

def average_image(img):
  return sum(img)*0.125

def av_thresh_av(img,thresh):
    img2 = subtract_av_thresh(img,thresh)
    img3 = average_image(img2)
    return img3

def open_close(img):
    img = ndimage.binary_opening(img)
    img = ndimage.binary_closing(img)
    return img

def bhoc_mask(slide):
    # Only need channel 8,9
    # Still have to change "fetch all" 
    img=slide.fetch_all()
    bh = img[9]-img[8] 
    return open_close( (bh>0.15) )

def imgoc_mask(img):
    img4 = av_thresh_av(img,0.15)
    return open_close(img4)



import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from scipy.ndimage import convolve1d
from scipy.ndimage import maximum_position

# Specialised filters
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import generic_laplace # overcomplex
from scipy.ndimage import sobel

# for plotting, shouldnt really be here
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
================================================================
    Image filters
================================================================
    Image processing filters used for non-ML preprocessing and preselection
"""



def img_convert_npy(npcube):
    """
    Converts image array to list of images. mainly for backwards compatability
    """
    return [ npcube[...,i] for i in range(npcube.shape[-1])]


def segmask_p2_p3_npy(npcube,back,halo):
    """
    Original segmentation stratergy doesnt work one new dirty data, all is cut
    WHY> bhoc mask doesnt work here. use simpler b-h
    """
    img = img_convert_npy(npcube)
    return open_close(pa_mask(img)*imgoc_mask(img)*bhoc_mask_npy(back,halo))

def segM2(raw,npcube,back,halo):
    """
    Imgoc mask works
    back * img[0] is STRONG , ie requires rotational hit AND natural pit
    img[0]*img[1] ie, requires correlation / unbroken light link
    # STrong, but kills too many real pits (try this + add together next in chain)

    AVERAGE! works powerfully, on both dirty and clean
    obv only works on raw data. ie without average subtracted, high recognition level
    """
    img = img_convert_npy(npcube)
    img2 = img_convert_npy(raw)
    av = sum(img2)
    y = [av*i for i in img]
    x = (img[2]*img[1]) +(img[2]*img[3])+(img[4]*img[3])
    x = x + (img[4]*img[5])+(img[6]*img[5])+(img[7]*img[6])+(img[1]*img[7])
    
    #z = gaussian_filter(back,sigma=10) # eg can filter for etchpit size
    z = gaussian_filter(av,sigma=2)
    z1 = gaussian_filter(back,sigma=3)

    z4 = gaussian_filter(back,sigma=3, order = 1) # Order of gaussian filter = derirative of gaussian
    # Produces cool 'radioactive' chart.

    z5 = gaussian_filter(back,sigma=3, order = 3)

    z2 = gaussian_filter(y[0],sigma=2) 
    z3 = gaussian_filter(y[1],sigma=2) 
    zL = [ gaussian_filter(yi,sigma=3) for yi in y ]
    t = (z*z > 0.1)
    t1 = z*t
    
    z7 = back*img[0]
    z8 = (img[2]*img[1])+(img[2]*img[3])+(img[4]*img[3])+(img[4]*img[5])
    """
    Gauss + laplace
    """ 
    zG = [  gaussian_filter(i,sigma=6) for i in img2 ]
    zG = sum(zG)
    zG = gaussian_laplace(zG,sigma = 6)

    """
    This is the cleanup image
    Geometric sum of image channels
    """
    z9 = img2[0]*img2[1]*img2[2]*img2[3]*img2[4]*img2[5]*img2[6]*img2[7]
 
    return zG  #  (z1 - z)**2 #z2*z3
    # geometric convoloution
    #img2[0]*img2[1]*img2[2]*img2[3]*img2[4]*img2[5]*img2[6]*img2[7]
    
    #(img[2]*img[1])+(img[2]*img[3])+(img[4]*img[3])+(img[4]*img[5])
    ##img = subtract_av_thresh(img,0.05)
    #p = PF_phase_R(img.copy())
    #a = PF_anti_R(img.copy())
    #prm = open_close(p)
    #arm = open_close(a)
    #return prm*arm*imgoc_mask(img)*(back > 0.15)
    #(back-halo) > 0.10 #bhoc_mask_npy(back,halo) #imgoc_mask(img)
    #open_close(prm*arm)


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

def subtract_av_thresh(img,threshold):
    av = average_image(img)
    img2=[]
    for i in img:
        img2.append(((i-av)>threshold)*(i-av))  
    return img2 

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

# TWO duplicate functions
def bhoc_mask_npy(back,halo):
    bh = halo-back 
    return open_close( (bh>0.15) )

def bhoc_mask(slide):
    # Only need channel 8,9
    # Still have to change "fetch all" 
    img=slide.fetch_all()
    bh = img[9]-img[8] 
    return open_close( (bh>0.15) )

def imgoc_mask(img):
    img4 = av_thresh_av(img,0.15)
    return open_close(img4)

def pa_mask(img):
    img_005 = subtract_av_thresh(img,0.05)
    p1r = PF_phase_R( img_005.copy() )
    a1r = PF_anti_R( img_005.copy() )
    prm = open_close(p1r)
    arm = open_close(a1r)
    return open_close( (prm*arm) )

#-------------------------------
# Phase filters

def PF(img,Sx,Sy):
  for i in range (0,8):
    #print(i)
    img[i] = np.roll(img[i],Sx[i],0)
    img[i] = np.roll(img[i],Sy[i],1)
  return img

def PF_phase(img):
    Ax = [-3,-5,-3,0]
    Ay = [-3,0,3,5]
    Bx = [3,5,3,0]
    By = [3,0,-3,-5]

    pSx = Ax+Bx
    pSy = Ay+By
    return average_image( PF(img,pSx,pSy) )

def PF_phase_R(img):
    Ax = [-3,-5,-3,0]
    Ay = [-3,0,3,5]
    Bx = [3,5,3,0]
    By = [3,0,-3,-5]

    pSx = Ax+Bx
    pSy = Ay+By
    pSx.reverse()
    pSy.reverse()
    return average_image( PF(img,pSx,pSy) )

def PF_anti(img):
    Ax = [-3,-5,-3,0]
    Ay = [-3,0,3,5]
    Bx = [3,5,3,0]
    By = [3,0,-3,-5]

    aSx = Bx + Ax
    aSy = By + Ay
    return average_image( PF(img,aSx,aSy) )

def PF_anti_R(img):
    Ax = [-3,-5,-3,0]
    Ay = [-3,0,3,5]
    Bx = [3,5,3,0]
    By = [3,0,-3,-5]

    aSx = Bx + Ax
    aSy = By + Ay
    aSx.reverse()
    aSy.reverse()
    return average_image( PF(img,aSx,aSy) )

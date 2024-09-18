
from dataset_tools import mhio
import os
import matplotlib.pyplot as plt
import numpy as np

def image_locations(foil_path = '/home/millward/Moedal_data/febdat/febdat_clean1/'):
    """
    Return x,y position Metadata for each image in a scan, as a Dict.
    key = image no.  value = (x,y) 
    
    True x,y position Metadata is stored in run directory.
    image_metadata = foil_path/Run#_image#_random#/Step_9_1.cfg
    
    - in file - 
    [Translation]
    X=4.409730
    Y=2.085791
    """

    full_path = foil_path + '{0:}/Step_9_1.cfg'
    print('Looking for metadata in: ',full_path)
    
    foldernames = os.listdir(foil_path)

    XY_pos = {}

    for fn in foldernames:    
        """ 
        Run#_image#_random# => run, slide, blah 
        """
        run,slide,blah = fn.split('_')
        #print(slide)
        f = open(full_path.format(fn),'r')
        f = [ ln.rstrip() for ln in f.readlines()]
        # get the right line
        x = float(f[1].strip('X='))
        y = float(f[2].strip('Y='))
        XY_pos[int(slide)] = (x,y)
        
    return XY_pos



def look_at_truexy():
    """
    The foil scans have the true xy data of the image locations in their metadata
    
    Tried same for x,y [position], basically exact same as [translation]
    """
    
    #foldernames = os.listdir('/home/millward/Moedal_data/febdat/febdat_clean1/')
    fin = '/home/millward/Moedal_data/febdat/febdat_clean1/'
    fin1 = '/home/millward/Moedal_data/febdat/febdat_clean1_reverse/'
    fin = '/home/millward/Moedal_data/febdat/febdat_dirty_reverse/'
    fin = '/home/millward/Moedal_data/febdat/febdat_dirty/'
    #fin = fin1+ '{0:}/Step_9_1.cfg'
    #foldernames = os.listdir(fin1)
    
    XY_pos = image_locations(fin1)
    
def plot_img_locations(XY_pos):
    X = [ XY_pos[key][0] for key in XY_pos.keys()] 
    Y = [ XY_pos[key][1] for key in XY_pos.keys()] 
   
    fig, ax = plt.subplots()
    ax.plot(X,Y,'ro')
    plt.show()
    


def generate_image_locations(plot=True,save = './Image_location_metadata.txt'):
    """
    Generates dictionary of image x,y locations from metadata
    Locations can be accessed via;
    
    image_locations[foil_key][slide#] = (x,y)  
    
    Note. JSON will convert interger slide keys to strings!
    """
    
    image_XYloc = {}

    foil_paths = {
        'c1' : '/home/millward/Moedal_data/febdat/febdat_clean1/',
        'c2' : '/home/millward/Moedal_data/febdat/febdat_clean2/',
        'cr' : '/home/millward/Moedal_data/febdat/febdat_clean1_reverse/',
        'd'  : '/home/millward/Moedal_data/febdat/febdat_dirty/',
        'dr' : '/home/millward/Moedal_data/febdat/febdat_dirty_reverse/'
             }

    for foil_key in foil_paths:
        image_XYloc[foil_key] = image_locations(foil_paths[foil_key])

    if plot:
        for foil_key in image_XYloc:
          plot_img_locations(image_XYloc[foil_key])

    if save != None:
        mhio.save_json(image_XYloc,save)
    
    return image_XYloc
    
    
def Xe_locations(plot=True,save = './Xe_metadata.txt'):
    """
    28 mar - 2021
    Doing repeat study w. Xe foil data. 
    """
    
    image_XYloc = {}
    
    foil_paths = {
        'Xe' : '/home/millward/Moedal_data/xe_10c_2020/Run1/',
        'Xer': '/home/millward/Moedal_data/xe_10c_2020/Run2/'
            }

    for foil_key in foil_paths:
        image_XYloc[foil_key] = image_locations(foil_paths[foil_key])

    if plot:
        for foil_key in image_XYloc:
          plot_img_locations(image_XYloc[foil_key])

    if save != None:
        mhio.save_json(image_XYloc,save)
    
    return image_XYloc

#Xe_locations()      
#generate_image_locations(plot=False)

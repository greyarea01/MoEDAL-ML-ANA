

import matplotlib.pyplot as plt
import numpy as np
from modules.dataset_tools import image_tools as it

def draw_circles(origin = (320,240),rad = 165,ec='r'):
    """
    Primary use is to plot circles over alignment pin-holes to indicate foil alignment
    - Same method can be modified to add bounding boxes
    """
    fig, ax = plt.subplots()
    ax.imshow(img,cmap='gray')
    circle_in  = plt.Circle(origin,rad, color=ec,fill=False) #// (x,y),r
    circle_out = plt.Circle(origin,rad+8, color=ec,fill=False)
    ax.add_patch(circle_in)
    ax.add_patch(circle_out)
    plt.show()


def draw_datum( foil='clean2_s',
                datum_points = [(1,322,242),(195,561,15),(351,600,248)],
                path='/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png'):
    """
    For a datum point, in (i=image# , x , y ) coords, draw 2 concentric circles, radius 195 pixels
    This should align with the alignment hole shown in the background image#
    """
    for datum in datum_points:
        bkg_image = it.Slide(foil,datum[0],path,'b').b
        draw_circles(origin=(datum[1],datum[2]),rad = 165)
        

"""
 Datum point alignment
 ---------------------------------------------------------------------
 Clean alignment circles appear in s1, s376/351,  s194/195/219/220
 
 Datum points;
  - Verified by inspection, using draw_datum()
  - Local ( i=image# , x_im , y_im ) coords
 
 clean 1 :  (1,320,240),  (195,563,15),  (351,602,247)
 clean 2 :  (1,322,242),  (195,561,15),  (351,600,248)
 clean r :  (1,322,240),  (195,74,12),  (351,36,246)
 
 dirty   :  (1,320,240),  (195,370,110),  (351,602,438)
 dirty r :  (1,321,244),  (195,72,16),  (351,38,246)
 
"""

datum_points = {
    'clean1_s'	:	[(1,320,240),  (195,563,15),	(351,602,247)],
    'clean2_s'	:	[(1,322,242),  (195,561,15),	(351,600,248)],
    'clean1_r_s':	[(1,322,240),  (195,74,12),		(351,36,246)],
    'dirty_s'	:	[(1,320,240),  (195,370,110),	(351,602,438)],
    'dirty_r_s'	:	[(1,321,244),  (195,72,16),		(351,38,246)],
    }

#for foil in datum_points: draw_datum(foil,datum_points[foil])

#foil = 'clean1_s'
#draw_datum2(foil,datum_points[foil])


#for foil in datum_points.keys():
    #print('\n Aligning: ',foil)
    #fsys = coordinates.Coord_system()
    #fsys.align_datum(datum_points[foil],True)
    #print('\n')





import numpy as np
from modules.dataset_tools import mhio

# Distance between 2 coordinate arrays
from scipy.spatial.distance import cdist


#// Path to retrive scanning metadata JSON
img_metadata_location = 'modules/dataset_tools/metadata/Image_location_metadata.txt'


"""
Author - l.millward@qmul.ac.uk
for MoEDAL experiment

-----------------------------------------------------------
 Transformation functions
-----------------------------------------------------------
"""

def distance(a,b):
    """
    Use np.subtract to cast both coordinates to numpy arrays
    This way result is agnostic over 2-tuples / vs 2-arrays
    
    """
    ds = np.subtract(a,b)
    return np.linalg.norm(ds,axis=-1)

def dist_matrix(points_a,points_b):
    """
    Distance between A=(N,2) B=(M,2) arrays of xy points
    """
    return cdist(points_a,points_b)

def translate(xy,dxy):
    """
    Casts coords to numpy, and adds them
    Tuple form would be tuple(np.add(xy,dxy))
    """
    return np.add(xy,dxy)
    
def invertX(xy):
    """
    Reflect the x-axis about y=0
    """ 
    return np.array((-xy[0],xy[1]))

def rotate(xy,phi):
    """
    2D rotation on global coordinates
    Cos(a)~1 sin(a)~a aproximation
    
    Important Y axis is negative, this will change the way rotations work
    """ 
    alpha = np.cos(phi)
    beta =  np.sin(phi)
    return (alpha*xy[0]-beta*xy[1],alpha*xy[1]+beta*xy[0])

    
def transform(XY,dxy=(0.,0.),phi=0.,invert_X=False):
    """
    Wrapper for combined translate / rotate / invert operations on coord list    
    """
    XY_new = [translate(xy,dxy) for xy in XY]
    XY_new = [rotate(xy,phi) for xy in XY_new]
    if invert_X: XY_new = [invertX(xy) for xy in XY_new]
    return XY_new
    
    
def theta_c(datum_xy):
    """
    Assuming a datum hole at (0,0), this will calculate a rotation angle
    such that the 3rd datum hole will intercept the x-axis.
    """
    Dx = datum_xy[0]
    Dy = datum_xy[1]
    theta_c = np.arctan( Dy/Dx )
    return theta_c   



class Coord:
    """
    Store etch-pit coordinates.
    'Local'  z =(i,x,y) tuple, where i = image number
    'Global' gxy = local_xy + image_location_xy (metadata)  
    
    """
    def __init__(self,z):
        self.i = z[0]
        self.xy = np.array([z[1],z[2]])
        self.sxy = 0
        self.gxy = 0
        
    # Access x,y as seperate attributes
    @property
    def x(self): return int(self.xy[0])
    @property
    def y(self): return int(self.xy[1])
    @property
    def z(self): return (self.i,self.x,self.y)

    def set_z(self,z):
        self.i = z[0]
        self.xy = np.array([z[1],z[2]])


def z_tuple(i,XY):
    return [(i,int(x),int(y)) for x,y in zip(*XY.T)] 


class Scan_grid:     
    """
    -------------------------------------------------------------
    Retrive image-scan metadata, and calculate relative positions
    xy_dict           ->  look-up 'xy' for image i 
    xy_array, index   ->  find nearest 'i' in array for query xy     
    -------------------------------------------------------------
    """
    def __init__(self,img_locations):
        """
        Init functions
        ----------------------------------
        1) Generate xy-dict from the meta-dat dictionary 'img_locations'
           Recentre origin, switch to LH cartesian system used for local xy
        2) Generate array of xy points, and corresponding slide number index
           This is used for finding nearest points.
        """ 
        self.xy_dict = {}      
        self.xy_array = []
        self.index = []            
                            
        for key,img_xy in img_locations.items():
            x = img_xy[0] - img_locations['1'][0] 
            y = img_locations['1'][1] - img_xy[1]                  
            self.xy_dict[int(key)] = np.array((x,y))
    
        for key, img_xy in self.xy_dict.items():
            self.index.append(key)
            self.xy_array.append(img_xy)
        self.xy_array = np.array(self.xy_array)
               
        
    def __getitem__(self,key):
        return self.xy_dict[key]
        
    def nearest(self,scan_xy):
        """
        Find closest image_point in xy array (k)
        The correspoinging image# is index[k]
        """
        dist = distance(scan_xy,self.xy_array)
        k = np.argmin(dist)
        return self.index[k]        
      

    
class Scan:
    
    def __init__(self,name,img_locations):
        self.name = name
        self.local_origin = np.array((320.,240.))
        self.origin = 0 # Location of first datum 
        self.theta_c = 0

        self.image_size = {'x':640,'y':480}        
        self.scangrid = Scan_grid(img_locations)
        self.pixelsize = 0.000254
              
        """
        Reverse scans are scans taken from the underside of the foil 
        (relative to normal orientation). Flipped scans are reverse scans where
        each image has been flipped in the x-axis to match the orientation
        of the corresponding normal orientation scan
        """
        self.reverse = False
        self.flipped = False
        

       
    """
    Local Coordinates <==> Scan Coordinates
    -----------------------------------------------------------
    """
            
    def scan_xy_to_ixy(self,scan_xy):
        """
        - Identify closest image location
        - IF it is within that image, subtract its location
        BIG issue: havent implemented that check yet!
        """
        i = self.scangrid.nearest(scan_xy)   
        xy = np.subtract(scan_xy,self.scangrid[i])
        if self.flipped: xy = invertX(xy)
        xy = (xy/self.pixelsize) + self.local_origin
        xy = xy.astype(int)
        return (i,xy[0],xy[1])
    
    
    def ixy_to_scan_xy(self,z):
        """
        Vector addition of image-origin to local x,y coordinates,
        after shifting and scaling the latter
        """
        local_xy = np.array((z[1],z[2]))
        local_xy = np.subtract(local_xy,self.local_origin)
        local_xy = self.pixelsize*local_xy
        if self.flipped: local_xy = invertX(local_xy)
        
        i = z[0]        
        img_xy = self.scangrid[i]
        return np.add(local_xy,img_xy)
        
    """
    Datum Alignment
    -----------------------------------------------------------
    """

    def align_datum(self,datum,verbose=False):
        datum = [Coord(z) for z in datum]
        
        """
        Origin alignment, with first datum[0] hole (X,Y,intercept)
        """           
        for d in datum: d.sxy = self.ixy_to_scan_xy(d.z)
        self.origin = datum[0].sxy
        for d in datum: d.gxy = translate(d.sxy,-self.origin)
        """
        Calculate the correction angle necessary to align the 3rd datum-hole
        With the x-axis: 
        """          
        self.theta_c = theta_c(datum[2].gxy)      
        for d in datum: d.gxy = self.get_gxy(d.z)
                                     
        if verbose:
            print('\tDatum\t:\tG_xy')
            for d in datum: print('{}\t:\t{}\t:\t{}'.format(d.z,d.sxy,d.gxy))
            print('Theta correction : ',self.theta_c)
            
    """
    Local Coordinates <==> Global Coordinates
    -----------------------------------------------------------
    """
    def get_gxy(self,z):
        sxy = self.ixy_to_scan_xy(z)
        gxy = rotate(translate(sxy,-self.origin),-self.theta_c)
        if self.reverse: gxy = invertX(gxy)       
        return np.array(gxy)/self.pixelsize
        
    def get_ixy(self,gxy):
        if self.reverse: gxy = invertX(gxy)
        gxy = np.array(gxy)*self.pixelsize
        sxy = translate(rotate(gxy,self.theta_c),self.origin)
        return self.scan_xy_to_ixy(sxy)
        
    def get_coord(self,z):
        co = Coord(z)
        co.gxy = self.get_gxy(z)
        return co
        
        
    """
    Utility
    -----------------------------------------------------------
    """     
      
    def image_borders(self,i,m=14.):
        """
        Locate the borders and margins of an image in global coordinates.
        useful for checking overlap between two image-scans
        """
        x = self.image_size['x']
        y = self.image_size['y']
        points = [(i,m,m),(i,x-m,m),(i,x-m,y-m),(i,m,y-m)]
        points = [self.ixy_to_scan_xy(z) for z in points]
        return points
        
        

    
class Coord_system:
    """
    -----------------------------------------------------------------
    Setup all coordinate systems used in Pb-NTD stack
    Alignment requires datum-point positions, and scanning metadata
    
    these are generated using one-time-scripts, and visual inspection
    
    Access via;
        CS = coordinates.Coord_system()
        CS['c1']
    -----------------------------------------------------------------
    """

    
    """
    Datum points obtained via visual inspection
    """
    datum_points = {
    'c1'	:	[(1,320,240),  (195,563,15),	(351,602,247)],
    'c2'	:	[(1,322,242),  (195,561,15),	(351,600,248)],
    'cr'    :	[(1,322,240),  (195,74,12),		(351,36,246)],
    'd'	    :	[(1,320,240),  (195,370,110),	(351,602,438)],
    'dr'	:	[(1,321,244),  (195,72,16),		(351,38,246)],
    }
    
    """
    Theta correction angle, observed values (rads)
    """
    theta_correction = {
    'c1' :  0.02524236876348675,
    'c2' : 0.025380239530464803,
    'cr' : -0.022387361188469013,
    'd'  :  0.022089650324572135, 
    'dr' : -0.022093113840123267,
    }
    
    """
    Default 'corrections' - When moving between 2 coord-systems it is
    sometimes neccesary to apply a small position offset. (Global coords)
    """
    offset = {  
        'cd':(-10,7),  # (-8,4) original
        'dc':(10,-7),
        'cdr':(-10,0), # (-12,4) earlier value
        'drc':(10,0),
        'crdr':(-7,9), # DOUBLE CHECK
        'null':(0,0), # Null conversion, 
            }
    

    def __init__(self,verbose=False):
        XY_metadata = mhio.load_json(img_metadata_location)  
        self.scans ={}
        for key in ['c1','c2','cr','d','dr']:
            self.scans[key] = Scan(key,XY_metadata[key])
            self.scans[key].align_datum(self.datum_points[key],verbose)

    def __getitem__(self,key):
        # Wrapper for accessing via index
        return self.scans[key] 
        
    def convert(self,z,preset):
        """
        Given a string preset (eg, 'cd') convert coordinates between the foils
        given in the preset string. This both converts between coordinate systems 
        and applies a beam angle of incidence correction to pit location
        
        Offsets calibrated for;
            'cd'   c1 -> d
            'ccf'  c1 -> cr / flipped
            'cdf'  c1 -> dr / flipped
        """ 
        
        """ c->d  d->c pair """
        def shift(key,scanin,scanout):
            gxy = self[scanin].get_gxy(z)
            gxy = translate(gxy,self.offset[key])
            return self[scanout].get_ixy(gxy)
               
        if preset == 'cd':
            return shift('cd','c1','d')
        
        if preset == 'dc':
            return shift('dc','d','c1')
        
        if preset == 'cdr':
            return shift('cdr','c1','dr')
        
        if preset == 'drc':
            return shift('drc','dr','c1')
        
        if preset == 'crdr':
            return shift('crdr','cr','dr')
        
        """ 
        Null conversions just transform between coordinate systems
        without applying a beam angle correction to pit location
        """
        if preset == 'cd0':
            return shift('null','c1','d')     
        if preset == 'cdr0':
            return shift('null','c1','dr')
        if preset == 'ccr0':
            return shift('null','c1','cr')
                    
        pass
    
    

class Overlap():
    """
    -------------------------------------------------
    Class for calculating overlap between two sets of XY
    points. 
    Distance between A=(N,2) B=(M,2) arrays of xy points
    self.dist is a (N,M) matrix
    -------------------------------------------------
    """
    def __init__(self,points_a,points_b,tolerance = 8.):
        self.dist = cdist(points_a,points_b)
        """
        # points, within tolerance 
        """
        self.match_a = np.count_nonzero(self.dist<tolerance,axis=1)
        self.match_b = np.count_nonzero(self.dist<tolerance,axis=0)
        
        """
        Was going to use:
            anb = a in b
            axb = a !in b  and vice versa, define mask arrays
        """
        #self.anb = (self.match_a > 0)
        #self.axb = (self.match_a == 0)
        
        """
        Count # points in the union and exclusion sets of each A/B pair
        """
        self.Nanb = np.sum((self.match_a > 0))
        self.Naxb = np.sum((self.match_a == 0))
        self.Nbna = np.sum((self.match_b > 0))
        self.Nbxa = np.sum((self.match_b == 0))
                
        """
        # index of nearest point in the other array
        """
        self.nearest_a = np.argmin(self.dist,axis=1)
        self.nearest_b = np.argmin(self.dist,axis=0)
        
        """
        Distance of the closest point in other array 
        """
        self.closest_dist_a = np.amin(self.dist,axis=1)
        self.closest_dist_b = np.amin(self.dist,axis=0)
        
        """
        Vector distance of closest point in other array. ie, seperate x,y result
        # 'BUGGED'
        """
        #self.closest_xy_dist_a = points_a-points_b[self.nearest_a]
        #self.closest_xy_dist_b = points_b-points_a[self.nearest_b]
        
    """
    Return array of indices where A has a matching partner in B, 
    and where it doesnt. + vice versa. Can then access via
    Pits_A[A_in_B]
    
    ISSUE! this return type wont work with anything other than a numpy array
    """
    def A_in_B(self):
        return self.match_a.nonzero()
        
    def A_not_B(self): 
        return np.where(self.match_a==0)
    
    def B_in_A(self): 
        return self.match_b.nonzero()
        
    def B_not_A(self): 
        return np.where(self.match_b==0)
    



def tally(pairs,show =False):
    """
    tally(pred_pits,true_pits,tolerance,show =False):
    ---------------------------------------------------------------
    Function for Comparing two ARRAYS of XY pit coordinates
    
    Take: 
      pred_pits = predicted locations
      true_pits = The true locations
          
    Return: 
      p_true  = Correct predictions (w. matching true pits)
      p_false = False positives
      
      t_pos = True example, correctly predicted
      t_neg = True example missed by predictions
      
    Due to clustering, fragmentation, acceptable location error etc..
    p_true != t_pos // ie, not a strict 1-1 matching between examples
    
    Likewise false negative rate is not relevant.
    ---------------------------------------------------------------                    
    """
    #pairs = Overlap(pred_pits,true_pits,tolerance)
    current = {}
    current['p_true']  = pairs.Nanb
    current['p_false'] = pairs.Naxb
    current['t_pos']   = pairs.Nbna
    current['t_neg']   = pairs.Nbxa
    
    #if show:
    #    for key,val in current.items(): print("\t{}\t{}".format(key,val))
    #return current


    pt = current['p_true']
    pf = current['p_false']
    pT = pt+pf
    tp = current['t_pos']
    tn = current['t_neg']
    tT = tp+tn

    print("         \t:   True/Pos \t: False/neg \t: Totals")
    print(" Predictions\t:\t{} \t:\t{}\t:\t{}".format(pt,pf,pT))
    print("   Truth \t:\t{} \t:\t{}\t:\t{}".format(tp,tn,tT))
    






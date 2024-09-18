import matplotlib.pyplot as plt
import numpy as np

"""
Author - l.millward@qmul.ac.uk
for MoEDAL experiment
"""



"""
-----------------------------------------------------
 Plotting functions
-----------------------------------------------------
"""

def plotD(Dict,invert=False,bkg=None):
    """
    Given a dictionary of desired icon, and a zipped XY object, will 
    Plot all items in dict. 
    eg, plotD({ 'b+': Cp.GXY,'rx': Cpr.GXY },invert=True)
    """
    fig, ax = plt.subplots()
    if bkg != None: ax.imshow(bkg,cmap='gray')
    if invert: ax.invert_yaxis()
    for key, XY in Dict.items():
        #print(XY)
        X,Y = zip(*XY)
        #print(X,Y)
        ax.plot(X,Y,key)
    plt.show()


def plot_local(i,
               pit_dict,
               foil='clean1_s',
               path='/home/millward/Moedal_data/febdat/png_nov/{0:}{1:}_{2:}.png'):
    """
    For a dictionary of objects to plot, in Z tuple format. Only plots objects
    That occur in the specified slide
    """
    bkg_image = it.Slide(foil,i,path,'b').b   
    fig, ax = plt.subplots() 
    ax.imshow(bkg_image,cmap='gray')
    """
    Assume list of local coords spanning entire foil
    """
    for key, pit_coords in pit_dict.items():
        X,Y = zip(*[(z[1],z[2]) for z in pit_coords if z[0]==i])
        ax.plot(X,Y,key)
    plt.show()

def compare(x,y,title,cmap1='gray',cmap2='gray'):
    """
    Given 2 image arrays x and y, plots side by side
    """
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(x,cmap1)
    ax2.imshow(y,cmap2)
    fig.suptitle(title)
    plt.show()

    
#def scangrid_XY
    
    
def display_pits(pil,index,rows=5,cols=10):
    
    # Take - automatic 'slice' of pil till rows*cols is filled
    i = index     
    fig, axes = plt.subplots(rows,cols)
        
    def plot_ax(ax):
        ax.imshow(pil[i],cmap='gray')
        ax.axis('off')
        ax.set_title(i,fontsize=8)     
        
    for j in range(rows): 
        for k in range(cols):
            plot_ax(axes[j,k])
            i +=1
      
    fig.suptitle('Pits: {} - to - {}'.format(index,i))
    plt.show()
    
    
    
    
def plot_histogram(x,title='Demo unit - see mh evaluate for sig/bkg hist'):
    """
    if used with multi-dim array will plot each one as a seperate catergory
    """
    xf = x.flatten()
    plt.hist(xf,bins='auto')
    plt.title(title)
    plt.show()
    
    
    

"""
-----------------------------------------------
 Image tools - plotting functions
 - Still in image-tools for now, to preserve stand-alone functionality
-----------------------------------------------
"""


"""
 Plotting: HOW TO?
 
 - Annotation
    for i, txt in enumerate(point_labels):
        ax.annotate(txt,(x[i],y[i]))
    # may require ax.scatter, or if using plt, plt.annotate

 - Markers
    . point
    o circle
    v ^ < > triangles
    1 2 3 4  triad (d,u,l,r)
    8 octogan
    s p h 8 square pent-hex-octo-gon
    X P thick x,+
    '$text_string$'
    
 - Color gradient
    use 'color map' rather than keyword
    plt.get_cmap("RdYlGn")  # optional
    
    col = np.arange(XY.shape[0])   #  steady gradient
    col = XY.T[2]                   # col from data
    
    ax.scatter(XY.T[0],XY.T[1],c=col) - may ned to use scatter rather than standard
    
 - General figure size
    plt.rcParams["figure.figsize"] = [16,9]
    changes defaults of all figures

"""



"""
--------------------------------------------------
 Matrix Plotting
 - Plot confusion matrices, etc
 - Plot 'foil view' over entire foil 

--------------------------------------------------
"""

def unit_test_matrix_plot():

    Y_labels = ['a','b','c']
    X_labels = ['A','B','C','D']

    test_matrix = np.array([[1.2,3.4,2.6,2.4],
                            [0.9,2.2,3.1,0.8],
                            [2.7,3.9,2.1,1.4]])

    matrix_plot(test_matrix,X_labels, Y_labels,' test title','xlabel','ylabel')

def unit_test_foil(xN=20,yN=25):
    """
    Displays a map of the standard foil-scan slide-layout
    """
    matrix = np.zeros((yN,xN),dtype=int)
    x_labels = np.arange(1,xN+1) 
    y_labels = np.arange(1,yN+1)
    for i in range(0,xN*yN):
        matrix[i%yN,int(i/yN)] = i+1
    matrix_plot(matrix,x_labels, y_labels,'Slide number','x_pos','y_pos')

def matrix_plot(matrix,X_labels=None,Y_labels=None,title=None,xtitle=None,ytitle=None):
    # TODO - Test is too big for 25 x 25, want interger no. display not float

    fig, ax = plt.subplots()

    """
    Create heatmap 
    """
    im = ax.imshow(matrix)

    """
    Set axis labels, first arguement sets position, second sets the label
    And other optional labels / titles
    """
    #if X_labels != None: plt.xticks(np.arange(len(X_labels)),X_labels )
    plt.xticks(np.arange(len(X_labels)),X_labels )
    #if Y_labels != None: plt.yticks(np.arange(len(Y_labels)),Y_labels )
    plt.yticks(np.arange(len(Y_labels)),Y_labels )
    if title != None: ax.set_title(title)
    if xtitle != None: plt.xlabel(xtitle)
    if ytitle != None: plt.ylabel(ytitle)
  
    for i in range(len(Y_labels)):
        for j in range(len(X_labels)):
            text = ax.text(j,i,matrix[i,j],ha="center",va="center",color="w")

    fig.tight_layout()
    plt.show()


def plot_pit_density(pit_density,xN=20,yN=25,imax = 250):
    """
    Plots pit densities over an entire or partial foil
    """
    
    matrix = np.zeros((yN,xN),dtype=int)
    x_labels = np.arange(1,xN+1) 
    y_labels = np.arange(1,yN+1)

    xpos = 0
    ypos = 0

    for i in range(0,imax-1):
        matrix[i%yN,int(i/yN)] = pit_density[i]



    matrix_plot(matrix,x_labels, y_labels,' HIP pit density','x_pos','y_pos')

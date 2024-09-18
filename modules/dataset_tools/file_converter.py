
import sys
import os

"""
Author - l.millward@qmul.ac.uk
for MoEDAL experiment

-----------------------------------------------------------------
	File conversion tools
-----------------------------------------------------------------
    Convert .tga files from the smartscope to png files 
"""

class image_converter:
    """
    takes input and output file string formats, and runs command line converstion
    os.system(convert filein.tga fileout.png)

    !Warning! - make sure file structure exists
    """

    def __init__(self,):

        #filein = '/ path to tga files /{Run#}_{stage#}_*/Step_{step#}_1.tga'
        self.filein = '/home/millward/Moedal_data/febdat/{0:}_{1:}_*/Step_{2:}_1.tga'
        #fileout = '/ path to png output/{foil#}_s{stage#}_{step#}.png'
        self.fileout = '/home/millward/Moedal_data/febdat/png_nov/{0:}_s{1:}_{2:}.png'

        self.convert = 'convert {0:} {1:} {2:}' # 1 = offset
        self.test = True # run conversion in sys, or just print line
        self.slides = range(1,501) #251)
        self.foil_dict = {
        'febdat_dirty/Run4'		: 'dirty',
        'febdat_dirty_reverse/Run1'	: 'dirty_r',
        'febdat_clean1/Run1'		: 'clean1',
        'febdat_clean2/Run1'            : 'clean2',
        'febdat_clean1_reverse/Run1'    : 'clean1_r',
            }

    def filein_fileout(self,filein,fileout):
        self.filein = filein
        self.fileout = fileout

    def conversion(self,key,slide,step,flip=False,offset=None):
        """
        Creates a conversion string
        """
        fin  = self.filein.format(key, slide, step)
        fout = self.fileout.format(self.foil_dict[key], slide, step)
        if flip==False and offset == None:
            return self.convert.format(fin,'',fout)
        if flip==True and offset == None:
            return self.convert.format(fin,' -flop ',fout)

    """
    Default conversions
    -------------------------------------
    """
    def png_from_tga(self,):
        """ Nominal conversion for multi-channel data, no flipping """
        for slide in self.slides:
            cs = [] # Conversion strings
            for key in self.foil_dict:
                for step in range(1,11):
                    cs.append(self.conversion(key,slide,step))
            if self.test:
                [print(string) for string in cs]
            else:
                [os.system(string) for string in cs]
            print('Completed slide:',slide)        
    
    

    def tga_to_png(self,):  
        """
        Default HARDCODED  tga->png conversion
        """
        flip=True
        
        for slide in self.slides:
            cs = [] #conversionStrings
            """
            Clean foils
            """
            cs.append(self.conversion('febdat_clean1/Run1', slide, 9))
            cs.append(self.conversion('febdat_clean2/Run1', slide, 9))
            cs.append(self.conversion('febdat_clean1_reverse/Run1', slide, 9,flip))

            """
            Exposed foils, all channels
            """
            for step in range(1,11):
               cs.append(self.conversion('febdat_dirty_reverse/Run1', slide, step,flip))
               cs.append(self.conversion('febdat_dirty/Run4', slide, step))
            
            if self.test == True:
                [print(string) for string in cs]
            else:
                [os.system(string) for string in cs]
            print('Completed slide:',slide)

    def overwrite(self,):
        """
        Use to overwrite conversions without a complete re-run
        eg, mistakes / corrupted files
        - Reverse should not have flip for comparing models on both sides
        """
        self.test = False
        flip = True
        for slide in self.slides:
            cs = [] #conversionStrings
            cs.append(self.conversion('febdat_clean1_reverse/Run1', slide, 9,flip))
            for step in range(1,11):
               cs.append(self.conversion('febdat_dirty_reverse/Run1', slide, step,flip))
            
            if self.test == True:
                [print(string) for string in cs]
            else:
                [os.system(string) for string in cs]
            print('Completed slide:',slide)

    def make_gifs(self,):
        # TODO - depreciated
        os.system('mkdir '+ self.outpath + '/gifs')
        for sn in range(1,26):
            filein = self.outpath + '/'+str(sn)+ '/*.png' 
            fileout = self.outpath + '/gifs/'+ str(sn)+ '.gif'
            convert = 'convert ' +filein + ' ' + fileout         
            if self.test == True: print(convert)
            if self.test == False: os.system(convert) 


def Xe_2020():
    Xe_convert = image_converter()
    Xe_convert.filein  = '/home/millward/Moedal_data/xe_10c_2020/{0:}_{1:}_*/Step_{2:}_1.tga'
    Xe_convert.fileout = '/home/millward/Moedal_data/Xe_png/{0:}_s{1:}_{2:}.png'
    Xe_convert.foil_dict = {
        'Run1/Run1' : 'Xe', 
        'Run2/Run2' : 'Xer',
          }
    Xe_convert.test=False
    Xe_convert.png_from_tga()
    
    





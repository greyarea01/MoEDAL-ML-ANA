import argparse
import sys
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import matplotlib.pyplot as plt

from dataset_tools import mhio

import pandas as pd

"""
=========================================================
    Dataset class
=========================================================

"""



class Dataset:
    """
    Custom dataset handler class for converting moedal datasets 
    (lists of pit np.arrays, sorted by catergory) into labelled TF-dataset objects 
    - handle K-Folding / test/train/val split
    """
    def __init__(self,):
    
        self.name = "Unnamed"
    
        # Fold options
        self.foldsize = 40
        self.n_folds = 5
        self.sig_folds = []
        self.bkg_folds = []
        # For self contained, validation set
        self.sig_validation = []
        self.bkg_validation = []
        # For higher stat plotting (biased)
        self.sig_all = []
        self.bkg_all = []
        # Batch options
        self.batch = 50
        self.repeat = 100
        self.shufflebuffer = 20000
        # Data options
        self.twoD = True

    """
    Options;
    - K folding
    """
         
    def print_defaults(self,):
        print( self.foldsize,'fold size')
        print( self.n_folds, 'n folds')

    def set_batchoptions(batch,repeat,shuffle):
        self.batch = batch
        self.repeat = repeat
        self.shufflebuffer = shuffle

    def make_train_set(self,signal,background):
        # return TF-ds from signal and bkg objects, 
        # batch/repeat/shuffle configurable, defaults for now
        A = ds_label(signal,(1.,),twoD=self.twoD)
        B = ds_label(background,(0.,),twoD=self.twoD)
        #A = ds_label(signal,(1.,0.),twoD=self.twoD)
        #B = ds_label(background,(0.,1.),twoD=self.twoD)
        dataset = A
        dataset = dataset.concatenate(B)
        dataset = dataset.repeat(self.repeat)
        dataset = dataset.shuffle(self.shufflebuffer)
        dataset = dataset.batch(self.batch)
        return dataset

    def make_test_set(self,signal,background):
        # Return TF-ds from list of signal and bkg objects 
        A = ds_label(signal,(1.,),twoD=self.twoD)
        B = ds_label(background,(0.,),twoD=self.twoD)
        #A = ds_label(signal,(1.,0.),twoD=self.twoD)
        #B = ds_label(background,(0.,1.),twoD=self.twoD)
        dataset = A
        dataset = dataset.concatenate(B)
        dataset = dataset.repeat().batch(len(signal)+len(background))#2*self.foldsize)
        # Use 2xFold size because there is a signal fold and a background fold
        return dataset

    def split_in_five(self,x):
        folds = []
        for i in range(self.n_folds):
            fold = x[i*self.foldsize:(i+1)*self.foldsize]
            folds.append(fold)
        return folds

    def get_folds(self,i,folds):
        test_fold = folds[i]
        train_fold  = []
        for j,fold in enumerate(folds):
            if (i != j):
                train_fold.extend(fold)
        return train_fold, test_fold

    def kfold(self,i):
        """
        return final test, train TF-datasets for a given fold
        """
        sig_train, sig_test = self.get_folds(i,self.sig_folds)
        bkg_train, bkg_test = self.get_folds(i,self.bkg_folds)  
        test  = self.make_test_set(sig_test,bkg_test)
        train = self.make_train_set( sig_train,bkg_train)
        return train,test

    def final(self,):
        """
        For use when K-Fold validation is finished
        use all training data, and use the validation data as the test set
        """
        test  = self.make_test_set(self.sig_validation, self.bkg_validation)
        train = self.make_train_set(self.sig_all,self.bkg_all)
        return train,test

    """
    Unit tests / Defaults
    """
    def load_defaults(self,):
        """
        Prepares dataset for t vs b classification 
        """
        self.twoD = True
        #s = pickle_load('/home/millward/Moedal/sig')
        b = mhio.pickle_load('/home/millward/Moedal/bottoms')
        t = mhio.pickle_load('/home/millward/Moedal/tops')
        # TODO SHOULD - shuffle so closely correlated pits dont appear in same fold
        self.sig_validation = b[:50]
        self.bkg_validation = t[:50]
        b = b[50:]
        t = t[50:]
        self.sig_folds = self.split_in_five(b)
        self.bkg_folds = self.split_in_five(t)
        
        
    def datalist(self,pits,shape=(28,28)):
        pitlist = []
        for p in pits: 
            if p.shape == shape: 
                pitlist.append(p)
        return pitlist



"""
=========================================================
    Dataset: examples
=========================================================

"""


class april_pb_pandas(Dataset):
    """
    Prototype - Using pandas dataframe as import source (vs pickle)
    data will contain ixy,gxy coordinate data, and channels as keys of data-frame
    Data will be partially incomplete - ie, not all entries will be correctly croped etc
    // Should this issue be dealt with earlier?
    //
    
    """
    
    def load(self,):
        cpits = pd.read_hdf('pits.h5','pits')
        dpits= pd.read_hdf('dpits.h5','dpits')
        drpits = pd.read_hdf('drpits.h5','drpits')
    
        # TEST load d and dr as sig and bkg sets
        crop28 = lambda x : (x.shape==(28,28))
        
        sig = []
        #b_exists = sig.apply(crop28)
        #sig = sig[b_exists]
        for s in dpits['b']:
            if crop28(s):
                sig.append(s) # np.flip(s,axis=1))      #s )
                
        print(len(sig))
        
        bkg = [] 
        #b_exists = bkg.apply(crop28)
        #bkg = bkg[b_exists]
        for b in drpits['b']:
            if crop28(b):
                bkg.append(b)
        
    
        self.sig_validation = sig[1000:1250]
        self.bkg_validation = bkg[1000:1250]
        
        self.foldsize = 200
        self.twoD = True
        
        self.sig_folds = self.split_in_five(sig)
        self.bkg_folds = self.split_in_five(bkg)
        self.sig_all = sig
        self.bkg_all = bkg
        
        
class april_pandas(Dataset):
    
    """
    Fail list contains identified fragments
    """
    fail_list = [27, 28, 30, 54, 55, 57, 60, 67, 73, 80, 87, 89, 93, 96, 97, 102, 116, 117, 130, 134, 163, 165, 172, 190, 215, 239, 255, 256, 259, 269, 272, 277, 282, 293, 332, 358, 406, 414, 443, 446, 447, 456, 458, 463, 480, 483, 486, 510, 514, 519, 539, 552, 561, 571, 573, 594, 636, 649, 702, 709, 713, 714, 738, 739, 759, 762, 786, 803, 817, 821, 825, 837, 844, 847, 904, 919, 928, 943, 978, 997, 1034, 1037, 1095, 1122, 1123, 1129, 1142, 1166, 1186, 1219, 1225, 1231, 1247, 1280, 1281, 1283, 1291, 1300, 1306, 1313, 1335, 1337, 1352]
    
    def load(self,channel):
        """
        Load sig and bkg as .h5 pandas dataframes. 'b' 'h' 'rim'
        """
        pathin = './pits/april_pandas/{}.h5'
        sig = pd.read_hdf(pathin.format('sig'))
        bkg = pd.read_hdf(pathin.format('bkg'))
    
        for i in [sig,bkg]: print(i['b'].shape)
    
        if channel == 'rim': shape=(28,28,8)
        else: shape = (28,28)
        
        sig = self.datalist(sig[channel],shape)
        bkg = self.datalist(bkg[channel],shape)
        
        for i in [sig,bkg]: print(len(i))
    
        #------------------------------------
    
        self.sig_validation = sig[1000:1250]
        self.bkg_validation = bkg[1000:1250]
        
        self.foldsize = 200
        if sig[0].shape == (28,28): self.twoD = True
        if sig[0].shape == (28,28,8): self.twoD = False
        
        self.sig_folds = self.split_in_five(sig)
        self.bkg_folds = self.split_in_five(bkg)
        self.sig_all = sig
        self.bkg_all = bkg  
        
    def load_not_failed(self,channel):
        
        pathin = './pits/april_pandas/{}.h5'
        sig = pd.read_hdf(pathin.format('sig'))
        bkg = pd.read_hdf(pathin.format('bkg'))
        fail_list = self.fail_list
        
        fail_list = np.array(fail_list)
        # Invert the index
        sig_pass = sig.iloc[~sig.index.isin(fail_list)]
        #print(sig_pass.head())
        
        """
        Sig pass should now only contain examples NOT in the fail list
        """
        for i in [sig_pass,bkg]: print(i['b'].shape)
        
        if channel == 'rim': shape=(28,28,8)
        else: shape = (28,28)
        
        sig_pass = self.datalist(sig_pass[channel],shape)
        bkg = self.datalist(bkg[channel],shape)
        
        for i in [sig_pass,bkg]: print(len(i))
    
        #------------------------------------
    
        """
        Only 1240 examples excluding fragments
        """
        self.sig_validation = sig_pass[1000:1240]
        self.bkg_validation = bkg[1000:1240]
        
        self.foldsize = 200
        if sig_pass[0].shape == (28,28): self.twoD = True
        if sig_pass[0].shape == (28,28,8): self.twoD = False
        
        self.sig_folds = self.split_in_five(sig_pass)
        self.bkg_folds = self.split_in_five(bkg)
        self.sig_all = sig_pass
        self.bkg_all = bkg 
        
class frag_dataset(Dataset):
    """
    Used for selecting / finding fragmentation events and weak ionisations
    """
    fail_list = [27, 28, 30, 54, 55, 57, 60, 67, 73, 80, 87, 89, 93, 96, 97, 102, 116, 117, 130, 134, 163, 165, 172, 190, 215, 239, 255, 256, 259, 269, 272, 277, 282, 293, 332, 358, 406, 414, 443, 446, 447, 456, 458, 463, 480, 483, 486, 510, 514, 519, 539, 552, 561, 571, 573, 594, 636, 649, 702, 709, 713, 714, 738, 739, 759, 762, 786, 803, 817, 821, 825, 837, 844, 847, 904, 919, 928, 943, 978, 997, 1034, 1037, 1095, 1122, 1123, 1129, 1142, 1166, 1186, 1219, 1225, 1231, 1247, 1280, 1281, 1283, 1291, 1300, 1306, 1313, 1335, 1337, 1352]
    
    def load(self,):
        pathin = './pits/april_pandas/{}.h5'
        sig = pd.read_hdf(pathin.format('sig'))
        bkg = pd.read_hdf(pathin.format('bkg'))
        
        """ Identify the fragments starting with the approximation that most false negatives were fragments (verified ~ 90%) """
        
        fail_list = self.fail_list

        fail_list = np.array(fail_list)
        sig_fail = sig.iloc[fail_list]
        bkg = bkg[:100]
        self.foldsize = 20

        sig = self.datalist(sig_fail['rim'],(28,28,8))
        bkg = self.datalist(bkg['rim'],(28,28,8))

        self.twoD = False
        self.sig_folds = self.split_in_five(sig)
        self.bkg_folds = self.split_in_five(bkg)
        
        
    def load_not_failed(self,):
        """ Load the pits which did NOT fail """
        pathin = './pits/april_pandas/{}.h5'
        sig = pd.read_hdf(pathin.format('sig'))
        bkg = pd.read_hdf(pathin.format('bkg'))        
        fail_list = self.fail_list

        fail_list = np.array(fail_list)
        # Invert the index
        sig_pass = sig.iloc[~sig.index.isin(fail_list)]
        #print(sig_pass.head())
        
        self.foldsize = 200
        self.twoD = False
        sig = self.datalist(sig_pass['rim'],(28,28,8))
        bkg = self.datalist(bkg['rim'],(28,28,8))
        
        print(len(sig),len(bkg))
        
        #self.sig_validation = sig[1000:]
        #self.bkg_validation = bkg[1000:]
        
        self.sig_folds = self.split_in_five(sig)
        self.bkg_folds = self.split_in_five(bkg)
        #self.sig_all = sig
        #self.bkg_all = bkg 

class feb_pb(Dataset):
    """
    Dataset using full 250 fs training images
    - 1306 double confirmed sig pits
    - alt 1439 available 'confirmed'
    - 1575 'fakes'
    First dataset using 'Etchpit' class. all info bundled into pit class
    """
    def load(self,):
    
        self.name = 'Feb Pb dataset'
    
        sig = mhio.pickle_load('./pits/feb_pits/sig_pits_1306')    
        bkg = mhio.pickle_load('./pits/feb_pits/bkg_pits_1575')

        sig = [p.rim for p in sig]
        bkg = [p.rim for p in bkg]

        # val = 300 / 1300, thus 5 folds of 200
        self.sig_validation = sig[1000:1300]
        self.bkg_validation = bkg[1000:1300]
        self.foldsize = 200
        sig = sig[:1000]
        bkg = bkg[:1000]
        self.twoD = False

        print('Dataset: ',self.name)
        print('Etch-pit shape = ', np.shape(sig[0]))
        
        self.sig_folds = self.split_in_five(sig)
        self.bkg_folds = self.split_in_five(bkg)
        self.sig_all = sig
        self.bkg_all = bkg




class nov_pbTest(Dataset):
    """
    818 double confirmed signal examples
    1575 background examples
    """

    def load(self,path='/home/millward/MLsept20/nov_pits/{0:}',twoD=False):
        self.twoD = twoD
        self.twoD = False
        tpits = mhio.pickle_load(path.format('confirmed_pits_fixed818'))
        fpits = mhio.pickle_load(path.format('fpits_1575'))
        # Stored as list of pit tuples, extract pit
        tpits = [ pit[3] for pit in tpits ]
        fpits = [ pit[3] for pit in fpits ]

        # val = 300 / 800
        self.sig_validation = tpits[500:800]
        self.bkg_validation = fpits[500:800]

        """
        Take 500 examples to use as training and testing sets via 5-fold cross-validation
        """
        tpits = tpits[:500]
        fpits = fpits[:500]
        self.foldsize = 100
        self.sig_folds = self.split_in_five(tpits)
        self.bkg_folds = self.split_in_five(fpits)
        self.sig_all = tpits
        self.bkg_all = fpits

    def test(self,):
        data = self.sig_validation
        lab = [1. for i in range(0,len(data)) ]
        #data = np.asarray(data)
        #print(np.shape(data))
        #data = tf.convert_to_tensor( data )
        dataset = tf.data.Dataset.from_tensor_slices( (data,lab))
        dataset.batch(6)
        return dataset
        

    def val(self,):
        A = ds_label(self.sig_validation,(1.,0.),twoD=False)
        B = ds_label(self.bkg_validation,(0.,1.),twoD=False)        
        return A.batch(300),B.batch(300)






class oct_pbTest(Dataset):
    """
    379 true examples
    504 fake examples

    real examples have been double confirmed by both clean foils with 10 pix tolerance
    slide 1,194,195 skipped
    fakes taken via frid from [200,197,196,191,190,173,169]

    dataset 'sanitized' via visual check of auto-selected pits 'clean'
    pits removed via sanitization = error in clean finder algo
    """

    def sanitize(self,pits,cull = [32,33,34,35,37,149,151,217,223,263,266,265]):
        return [i for j, i in enumerate(pits) if j not in cull]

    def load(self,path='/home/millward/MLsept20/oct_pits/{0:}',twoD=False):
        self.twoD = twoD
        self.twoD = False
        true_pits = mhio.pickle_load(path.format('pits'))
        fake_pits = mhio.pickle_load(path.format('fakepits'))

        true_pits = self.sanitize(true_pits) # 367 remain
        """
        Take 100 examples as validation set
        """
        self.sig_validation = true_pits[250:350]
        self.bkg_validation = fake_pits[250:350]
        """
        Take 200 examples to use as training and testing sets via 5-fold cross-validation
        """
        true_pits = true_pits[:250]
        fake_pits = fake_pits[:250]
        print(np.shape(true_pits[0]))
        for pit in true_pits:
            print(np.shape(pit))
        self.foldsize = 50 # vs default 40
        self.sig_folds = self.split_in_five(true_pits)
        self.bkg_folds = self.split_in_five(fake_pits)
        self.sig_all = true_pits
        self.bkg_all = fake_pits

class june_pbTest(Dataset):
    """
    250 signal + bkg examples from Pb test beam + LHC
    'Signal' = Confirmed Pb ion beam
    Data is list [] of etch pits, each 'pit' is a 28*28*8 array 
    """
    def load(self,path,twoD=False):
        self.twoD = twoD
        true_pits = mhio.pickle_load(path.format('pits_fixed297'))
        fake_pits = mhio.pickle_load(path.format('fakepits_fixed288'))
        """
        Take 50 examples as validation set
        """
        self.sig_validation = true_pits[200:250]
        self.bkg_validation = fake_pits[200:250]
        """
        Take 200 examples to use as training and testing sets via 5-fold cross-validation
        """
        true_pits = true_pits[:200]
        fake_pits = fake_pits[:200]
        self.sig_folds = self.split_in_five(true_pits)
        self.bkg_folds = self.split_in_five(fake_pits)

    def load2(self,path='/home/millward/ML_june/IT_june/pits/{0:}'):
        self.twoD = False
        true_pits = mhio.pickle_load(path.format('dpits'))
        fake_pits = mhio.pickle_load(path.format('dfakepits'))
        #print(len(true_pits))
        """
        Sanitize the dataset by deleting poor examples
        """
        sanitize1 = [16,17,18,19,20,21,22,41,54,60,85,94,99,102,104,105,126,128,130,131]
        sanitize2 = [136,159,161,164,171,182,183,192,193,195,198,199]
        sanitize3 = [200,201,202,210,213,220,236,241,246,254,256,270,282,283,284,286,287]
        #true_pits = np.delete(true_pits,sanitize3)
        #true_pits = np.delete(true_pits,sanitize2)
        #true_pits = np.delete(true_pits,sanitize1)
        sanitize = sanitize1+sanitize2+sanitize3
        true_pits = [i for j, i in enumerate(true_pits) if j not in sanitize]

        #print(len(true_pits)) # 248 examples after sanitization

        self.sig_validation = true_pits[200:248]
        self.bkg_validation = fake_pits[200:248]
        """
        Take 200 examples to use as training and testing sets via 5-fold cross-validation
        """
        true_pits = true_pits[:200]
        fake_pits = fake_pits[:200]
        self.sig_folds = self.split_in_five(true_pits)
        self.bkg_folds = self.split_in_five(fake_pits)
        self.sig_all = true_pits
        self.bkg_all = fake_pits


    def val2(self,):
        A = ds_label(self.sig_validation,(1.,0.),twoD=False)
        B = ds_label(self.bkg_validation,(0.,1.),twoD=False)
        return A.batch(48), B.batch(48)
        #return A.batch(97), B.batch(88)




    def val(self,):
        true_pits = mhio.pickle_load('/home/millward/ML_sept_jan/mayPits/pits_fixed297')
        fake_pits = mhio.pickle_load('/home/millward/ML_sept_jan/mayPits/fakepits_fixed288')
        A = ds_label(true_pits[200:250],(1.,0.),twoD=False)
        B = ds_label(fake_pits[200:250],(0.,1.),twoD=False)
        #A = ds_label(true_pits[200:297],(1.,0.),twoD=False)
        #B = ds_label(fake_pits[200:288],(0.,1.),twoD=False)
        return A.batch(50), B.batch(50)
        #return A.batch(97), B.batch(88)

    # TODO - quick hack
    def true_pits(self,):
        true_pits = mhio.pickle_load('/home/millward/ML_sept_jan/mayPits/pits_fixed297')
        return np.asarray( true_pits[200:250] )

    def fake_pits(self,):
        fake_pits = mhio.pickle_load('/home/millward/ML_sept_jan/mayPits/fakepits_fixed288')
        return np.asarray( fake_pits[200:250] )


    def pits(self,):
        pits = mhio.pickle_load('/home/millward/ML_sept_jan/mayPits/pits')
        t = mhio.pickle_load('/home/millward/ML_sept_jan/mayPits/fakepits')
        A = ds_label(pits[:250],(1.,0.),twoD=False)
        B = ds_label(t[:250],(0.,1.),twoD=False)
        dataset = A
        dataset = dataset.concatenate(B)
        #dataset = dataset.repeat(50)
        #dataset = dataset.shuffle(20000)
        #dataset = dataset.batch(50)
        dataset = dataset.batch(500)
        return dataset



#--------------------------------------------------------------

class validation:
    def __init(self,):
        self.s 
        self.b
        self.t

    def b(self,):
        b = pickle_load('/home/millward/Moedal/bottoms')
        A = ds_label(b[:50],(1.,0.))
        A = A.batch(50)
        return A

    def t(self,):
        t = pickle_load('/home/millward/Moedal/tops')
        A = ds_label(t[:50],(0.,1.))
        A = A.batch(50)
        return A

    def s(self,):
        s = pickle_load('/home/millward/Moedal/sig')
        A = ds_label(s[:50],(0.,1.))
        A = A.batch(50)
        return A


#------------------------------------------------------------

def make_label(x,l): 
    #return np.full(x.shape[0],l) # Doesnt handle generic label shapes
    lab = [l for i in range(x.shape[0])]
    return np.asarray(lab)

def ds_label(x,l,twoD=True):

    data = np.asarray(x) #.astype('float32')
    # Not working in tf 2.2
    #print('prior shape',data.shape)

    if twoD == True:     
        #data = data[...,0] originally used for selecting 1/8 channels
        data = np.expand_dims(data,axis=-1)
        #print('expanded shape',data.shape)
     
    labels = make_label(data,l)
    data = tf.convert_to_tensor( data )
    labels = tf.convert_to_tensor( labels )
    dataset = tf.data.Dataset.from_tensor_slices((data,labels))
    return dataset
 

#-------------------------------------------------



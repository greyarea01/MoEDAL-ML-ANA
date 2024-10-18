
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import tensorflow as tf
from tensorflow import keras

import numpy as np
import random

from dataset_tools import mhio

from . import cnn
from . import dataset
from . import history

FLAGS = None

"""
==============================================================
	ML4prototype - edited from ML3_new
	last edit 30 aug
	
	input 
	- data should be prelabelled, AND prefolded (5), and pre-split
	- suggest test/train via k-fold, + seperate 5 fold cross-validation for val
	- hyper-params

	output
	- training logs
	- trained model
	- settings required to duplicate result

	Change fundemental structure 
	- run train loop over all aspects of a model and explitly return relevant info
	# Could use decorator to specify model
"""



class Meta_Run:

    def __init__(self,DATA,CNN,dirpath = './MLout/test'):
        self.dirpath = dirpath
        self.data = DATA
        self.MODEL = CNN
        self.hists = []
        
        os.mkdir(dirpath)
        os.mkdir(dirpath+'/models')
        os.mkdir(dirpath+'/plots')
        
        names = ['fold-1','fold-2','fold-3','fold-4','fold-5']
        mpath = dirpath + 'models/{0:}.h5'
        for i in range(0,5):        
            print('Running on fold:',i+1)
            train,test = self.data.kfold(i)
            hist = train_model(
                train,
                test,
                names[i],
                model=self.MODEL.create_model(),
                mpath=mpath.format(names[i])
                )        
            self.hists.append(hist)
            
        mhio.pickle_dump(self.hists,dirpath+'/hists')
        
        save = dirpath+'/plots/'
        
        for hist in self.hists:
            save = dirpath+'/plots/'
            hist.plot_acc(val=True,show=False,save=save)
            hist.plot_loss(val=True,show=False,save=save)
            
        history.plot_histAverage(self.hists,'acc',show=False,save=save)
        history.plot_histAverage(self.hists,'loss',show=False,save=save)




def train_model(train,test=False,name='test',hyp=None,model=None,mpath=None):
    
    if mpath is None:
        model_path = './MLout/scratch/{0:}.h5'.format(name)
    else: model_path = mpath.format(name)

    if model == None:
        network = cnn.Cnn()
        model = network.create_model()
        model.summary()
        model.run_eagerly = True

    if hyp==None:
        hyp = default_params()

    if test==False: 
        test=None
        val_steps = None
    else:
        val_steps = 1
        
    run = model.fit(train,
                steps_per_epoch = hyp["steps_pe"], 
                epochs=hyp["epochs"],
                validation_data = test,
                validation_steps = val_steps,) 

    print(run.params)

    hist = history.Run_History(run.history,name=name,m_path=model_path,hyp=hyp)
    #,kparams = run.params)

    model.save(model_path)

    # Release model memory
    del model
    return hist


def train_model_tf19(train,test,name=None,model=None,hyp=None):
    """
    TF 1.9 code
    """
    # TODO - save model by name / run directory
    opt = cnn.adam(0.00005) #0002
    # model_path = './MLout/model_test.h5'
    model_path = './MLout/{0:}.h5'.format(name)

    if model==None:
        #model = mh.create_model( mh.keras_test_model4_tanh(),opt )
        model = cnn.create_model( cnn.keras_test_model4_tanh(),opt )

    if hyp==None:
        hyp = default_params()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        run = model.fit(train,
                    steps_per_epoch = hyp["steps_pe"], 
                    epochs=hyp["epochs"],
                    validation_data = test,
                    validation_steps = 1,) 
        if model_path != None: 
            model.save(model_path)
        hist = history.Run_History(run.history,name,model_path,hyp)

        return model,hist


#------------------------------------------------------------

def default_params():
    # batch-size dictated in dataset, not alterable here
    hyp = { "batch_size": 100,   # Examples passed, before updating weights - 50 good
     	    "epochs": 60,       # Times to repeat the entire Dataset  - 50 good }
	    "steps_pe": 5      # learning steps in each epoch, - 20 good, :. epochs = k examples 
           }
    return hyp

#-------------------------------------------------


# TODO - 1 line functs pointless
def SaveModel(model,path):
    # Save keras model and weights, architecture, should be HDF5, .h5 whether model or weights
    model.save(path)
    #model.save_weights(path)
 
def LoadModel(path):
    # Load keras model and weights, architecture
    # also, loss, optimiser
    return keras.models.load_model(path)
    # model.load_weights(path)

def make_dir(path):
    os.system('mkdir '+path)



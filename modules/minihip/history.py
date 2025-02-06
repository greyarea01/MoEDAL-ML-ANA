
import argparse
import sys
import os
import tensorflow as tf
from tensorflow import keras

import numpy as np
import random
import matplotlib.pyplot as plt

import json
import pickle


"""
=========================================================
    Run History class
=========================================================

"""

class Run_History:

    def __init__(self,history,name=None,m_path=None,hyp=None,kparams=None):
        # self.name  // Name of this run, eg 'fold-1'
        self.name = name
        self.history = history
        self.keys = list(history.keys())
        self.model_path = m_path
        # self. Data Used
        self.hyperparameters = hyp
        
        # Older TF version does not have the metrics key
        if kparams is not None:    
            self.kparams=kparams
            if 'metrics' in kparams:
                self.metrics = self.kparams.pop('metrics')

    # def save_as_pickle
    # def save_json
    # def load json
    # loading pickle brings up loading class issue
    #def save_run
    
    def plot_metric(self,metric='acc',val=False,show=True,save=False):
        """
        Plot generically named metric
        """
        plt.plot(self.history[metric], label='Training')
        val_key = 'val_'+metric
        if val is True: plt.plot(self.history[val_key] ,label='Validation')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.title(metric)
        # Add Title

        plt.legend()
        if self.name != None: 
            plt.title( self.name + ' - {}'.format(metric))
        if save != False: 
            plt.savefig(save+self.name+'_{}.pdf'.format(metric)) 
        if show is True:
            plt.show()
        else: plt.close()
        
        
    def plot_acc(self,val=False,show=True,save=False):
        plt.plot(self.history['acc'], label='Training')
        if val is True: plt.plot(self.history['val_acc'] ,label='Validation')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        # Add Title

        plt.legend()
        if self.name != None: plt.title( self.name + ' - Accuracy')
        if save != False: plt.savefig(save+self.name+'_acc.pdf')
        if show is True: plt.show()
        else: plt.close()

    def plot_loss(self,val=False,show=True,save=False):
        plt.plot(self.history['loss'], label='Training')
        if val is True: plt.plot(self.history['val_loss'] ,label='Validation')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # Add Title
        plt.legend()
        if self.name != None: plt.title( self.name + ' - Loss')
        if save != False: plt.savefig(save+self.name+'_loss.pdf')
        if show is True: plt.show()
        else: plt.close()

    def print_hyperparameters(self,):
        # TODO - change to string formatting
        print(self.name)
        print(self.model_path)
        print('#####################################')
        print('keras RUN parameters: \n',self.kparams)
        print('metrics: \n',self.metrics)
        for key in self.hyperparameters:
            print( key , self.hyperparameters[key] )


"""
------------------------------------------------
    Plotting
------------------------------------------------
"""

def plot_histAverage(hists,metric='acc',show=True,save=False):
    #for hist in hists:
    #    hist.plot_acc(val=False,show=False)
    #plt.show()  
    
    av_acc = average_lists( [hist.history[metric] for hist in hists] )
    av_acc = average_lists( [hist.history['val_'+metric] for hist in hists] )
    plt.ylabel('average '+metric)
    plt.xlabel('epoch')
    plt.title('Ensemble - average {}'.format(metric))
    if save != False: plt.savefig(save+'average_'+metric+'.pdf')
    if show is True: plt.show()
    else: plt.close()


def average_lists(list_of_lists):
    """
    works to calc mean and sd for k-fold
    """
    x = np.array([np.array(list) for list in list_of_lists])
    x_av = np.average(x,axis=0)
    x_er = np.std(x,axis=0)
    # plot w. error bars:    plt.errorbar(x,y,e)
    plt.errorbar(np.arange(x_av.shape[0]),x_av,yerr=x_er,label='Accuracy - Avg.')
    #plt.show()


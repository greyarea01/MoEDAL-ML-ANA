import argparse
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from . import cnn, dataset, history


def predictions(data,model_name,score=True):
    """
    Model loading is bugged, has to be loaded WITHIN the sess and init loop
    otherwise weights get reset
    2nd workaround: use load_weights('model.h5') to reset correct weights 
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model = keras.models.load_model(model_name)
        predictions =  model.predict(data,verbose=1, steps = 1)         
        sess.close()

        if score==True:
            # Just first value reflects the sig score
            return predictions[...,0]
        else:
            return np.argmax(predictions,1)



"""
------------------------------------------------
	Evaluation metrics
------------------------------------------------
	- ROC curve
	Plot signal efficiency and background efficiency 

	- Histogram
	Plot S/B seperation and overlap

	- 5-fold validation
	- 5-fold training history
	- confusion

"""

#----------------------------------------------
# TODO - roc and histogram scores should be normalised to the 0-1 range
# TODO - overide histogram for multiple classes

def ROC_curve(sig,bkg,bins=1000,show=True,save=False,noClose=False):
    """
    Accepts list or array of signal scores
    And background scores. Bins approx resoloution 
    BUG - ROC not normalised
    TODO: add area under curve (and norm)
    """
    x,y = [],[]
    for i in np.linspace(0,1,bins):
        tpr,fpr = (np.sum(sig > i)/(sig.size*1.)),(np.sum(bkg < i)/(bkg.size*1.) )
        y.append(tpr),x.append(fpr)
    plt.plot(x,y)
    plt.ylabel('Sig efficiency')
    plt.xlabel('Bkg rejection')
    plt.title('ROC curve')
    # For plotting multiple ROC's
    if save != False: plt.savefig(save+'_ROC.pdf')
    if show is True: plt.show()
    elif noClose==False: plt.close()

def Histogram(sig,bkg,bins=15,show = True,save=False,noClose=False): #,normed = 0):
    """
    Accepts list or array of signal scores
    And background scores
    """
    #plt.hist(sig,bins, normed=normed, lw = 1, alpha = 0.5, label='Signal')
    #plt.hist(bkg,bins, normed=normed, lw = 1, alpha = 0.5, label='Background')
    plt.hist(sig,bins,  lw = 1, alpha = 0.5, label='Signal')
    plt.hist(bkg,bins,  lw = 1, alpha = 0.5, label='Background')
    plt.xlabel('Signal-Score')
    plt.ylabel('Examples')
    plt.legend()
    plt.title('Classification scores Histogram')
    # for plotting multiple histos
    if save != False: plt.savefig(save+'_Histogram.pdf')
    if show is True: plt.show()
    elif noClose==False: plt.close()

def unit_test_histogram():
    s = np.array([0.98,0.97,0.76,0.6,0.54])
    b = np.array([0.21,0.12,0.46,0.56,0.34])
    Histogram(s,b)




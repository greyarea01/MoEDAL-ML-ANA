
import tensorflow as tf
from tensorflow import keras
#import keras
#from keras import layers
from tensorflow.keras import layers
import numpy as np

"""
Author - l.millward@qmul.ac.uk
for MoEDAL experiment
"""


def make_inference_model( path='./MLout/fold-1.h5',
                          layercut=10,shape=(480,640,8),
                          verbose='False'):
    """
    -----------------------------------------------------------------------------
    - Take existing CNN model and converts to FCNN with reduced output dims.
    - Transfer layers up until 'layer-cut' into a new model with input 'shape'.
    - Assumes no retraining necessary, compiles w generic metrics.
    -----------------------------------------------------------------------------
    """
    model = keras.models.load_model(path)

    if verbose:
        print('Original model:',path)
        model.summary()

    model2 = keras.Sequential()
    model2.add(keras.Input(shape=shape))
    for layer in model.layers[:layercut]: 
        model2.add(layer)
    model2.compile('adam',loss = 'binary_crossentropy',metrics=['acc']) 

    if verbose:
        print('new model:')
        model2.summary()

    return model2


class Ensemble:
    """
    -----------------------------------------------------------------------------
    From a set of CNN clasifier models produce an ensemble of FCNN models that
    can produce a set of prediction heatmaps for a given slide image 
    -----------------------------------------------------------------------------
    """
    def __init__(self,path='./MLout/fold-{0}.h5',layercut=9,shape=(480,640,8)):
        self.path = path
        self.layercut = layercut
        self.shape = shape
        self.models = []

        """
        Assumes standard input of 5 cnn models
        """
        for i in range(1,6):
            #model = make_inference_model(self.path.format(i),
            #                             self.layercut,
            #                             self.shape)
            self.models.append(make_inference_model(self.path.format(i),
                                     self.layercut,
                                     self.shape))

    def predict_slide(self,x):
        """
        Assume (h,w,c) input, whereas (n,h,w,c) is required
        output is (h,w,model#)
        """
        x = np.expand_dims(x,axis=0) 
        print(np.shape(x))
        y = [ model.predict(x,verbose=0, steps = 1)[0] for model in self.models ]
        print(np.shape(y[0]))
        return np.concatenate(y,axis=-1)


#def get_models(path = './MLout/scratch/f{0}r.h5',layercut=10,shape=(480,640,8) )

def get_model(path,input_layer,layercut):
    model = keras.models.load_model(path)
    model2 = keras.Sequential()
    model2.add(input_layer)
    for layer in model.layers[:layercut]: 
        model2.add(layer)
    #for layer in model2.layers:
    #    layer.trainable = False
    model2.trainable = False
    return model2


def FCNN_retrain():
    
    """"conv2d_8
    3 Input data channels (10 total)
    """
    rim_input = keras.Input(shape=(480,640,8))
    b_input = keras.Input(shape=(480,640,1))
    h_input = keras.Input(shape=(480,640,1))
    
    #model_name_list
    rpath = './MLout/scratch/f{0}r.h5'
 
    rim_models = []
    for i in range(1,6):
        model = get_model(rpath.format(i),rim_input,9)
        """ change name so each model is unique, needed so each fcnn layer is unique """
        #model._name = 'm{}'.format(i)
        for layer in model.layers:
            layer._name = layer._name + '_{}'.format(i)
        # layer._name = layer.name + str('asdf')
        rim_models.append(model)
        
    outputs = [model.output for model in rim_models]
    fc_inputs = layers.Concatenate()([model.output for model in rim_models])
    
    x = layers.Dense(10, activation = 'relu')(fc_inputs)
    x = layers.Conv2D(30, kernel_size = (2,2), padding='same',activation='relu')(x)
    x = layers.Dense(20, activation = 'relu')(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)
    
    fcnn = keras.Model(inputs = rim_input, outputs = x)
    fcnn.compile('adam',loss = 'binary_crossentropy',metrics=['acc']) 
    fcnn.summary()
    
    return fcnn
    
    
    
    
    
    
    
    
    
    
    


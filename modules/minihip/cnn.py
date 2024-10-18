import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#from tensorflow.python.keras.layers import Conv2D, Input, SeparableConv2D, Conv3D
import numpy as np

"""
=========================================================
    CNN class
=========================================================
    - for Tensorflow 1.9

    Layers Ref;

    layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='relu')
    layers.MaxPooling2D(padding='same')
    layers.Dense( 10, activation = 'relu')

      
    keras metrics ref;

      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),



    Warning!
    Softmax does not play nice with any form of single depth output 
    eg, fcns

    Warning!
    Bug when using logit output and softmax -> get 50% acc w no learning
    prob due to mixing and matching wrong label type in dataset

"""



def k_sequential():
    """
    Default model used in Xe Ion study
    Works consistently for Xe/Pb, 3D/2D (28,28,1), robust wrt h-params 
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28,8))) # specifying input 
    model.add(layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.GlobalMaxPooling2D()) # compress to 1 spatial dimension
    model.add(layers.Dense(22, activation = 'relu'))
    model.add(layers.Dense(2, activation = 'softmax'))
    #model.summary()
    return model

def k_sequential2():
    mylayers = [ 
    keras.Input(shape=(28,28,8)),
    layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'),
    layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'),
    layers.GlobalMaxPooling2D(),
    layers.Dense(2, activation = 'softmax')
    ]
    model = keras.Sequential(mylayers)
    return model

def k_sequential_dec():
    """
    Default model used in Xe Ion study
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28,8))) # specifying input 
    model.add(layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (3,3), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.GlobalMaxPooling2D()) # compress to 1 spatial dimension
    model.add(layers.Dense(22, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    #model.summary()
    return model


"""
--------------------------------------------------------------------------------------
	Experimental Models
--------------------------------------------------------------------------------------
"""

def k_transferTest():
    """
    28,28,8 -> 1 output
    global maxpool reserved till end,
    running without this generates inference heatmap
    """

    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28,8))) # specifying input 
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    #model.add(layers.MaxPooling2D(padding='same'))

    model.add(layers.Dense(22, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    # runs into issue if the final global max pool layer isnt present somewhere
    model.add(layers.GlobalMaxPooling2D())
    #model.summary()
    return model


def ktt_functional():
    """
    Functional classification model
    Use keras functional API
    """
    input = keras.Input(shape=(28,28,8))
    x = layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh')(input)
    x = layers.MaxPooling2D(padding='same')(x)
    x = layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh')(x)
    x = layers.MaxPooling2D(padding='same')(x)
    x = layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh')(x)
    x = layers.MaxPooling2D(padding='same')(x)
    x = layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh')(x)
    fc = layers.Dense(22, activation = 'relu')(x)
    fc_out = layers.Dense(1, activation = 'sigmoid')(fc)
    
    """
    Fully convoloutional model with 1 output depth can be used if final maxpooling layer is omitted
    """
    fc_model = keras.Model(input, fc_out, name="fc_model")

    """
    CNN classifier with single output formed by final global pooling layer
    """
    class_out = layers.GlobalMaxPooling2D()(fc_out)
    ktt_classifier = keras.Model(input, class_out, name="cnn_ktt")
    return ktt_classifier


def pit_class_2d():
    """
    28,28,8 -> 1 output
    global maxpool reserved till end,
    running without this generates inference heatmap
    """

    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28,1))) # specifying input 
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    #model.add(layers.MaxPooling2D(padding='same'))

    model.add(layers.Dense(22, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    # runs into issue if the final global max pool layer isnt present somewhere
    model.add(layers.GlobalMaxPooling2D())
    #model.summary()
    return model


def pit_Nclass_2d(N=1):
    """
    28,28,8 -> 1 output
    global maxpool reserved till end,
    running without this generates inference heatmap
    """

    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28,1))) # specifying input 
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    model.add(layers.MaxPooling2D(padding='same'))
    model.add(layers.Conv2D(32, kernel_size = (4,4), padding='same',activation='tanh'))
    #model.add(layers.MaxPooling2D(padding='same'))

    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(22, activation = 'relu'))
    model.add(layers.Dense(N, activation = 'softmax'))

    #model.summary()
    return model




def conv3d():
    """
    prototype model for 3d-convoloution
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28,8))) # specifying input 
    model.add(layers.Reshape((28,28,8,1)))
    
    # 2x 3D conv+MP repeating unit
    model.add(layers.Conv3D(filters=32, kernel_size=3,padding='same',activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3,padding='same'))
    
    model.add(layers.Conv3D(filters=32, kernel_size=3, padding='same',activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3,padding='same'))
    
    # Simple Dense output
    model.add(layers.Dense(22, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    # Shape is now (4,4,1,1)
    model.add(layers.Reshape((4,4,1)))
    # runs into issue if the final global max pool layer isnt present somewhere
    model.add(layers.GlobalMaxPooling2D())
    model.summary()
    # RESULT 95% accuracy
    return model


class Cnn:
    """
    Prototype for cnn class
    Build and return CNN kmodel class for a given architecture
    repeatrable for different folds / runs
    """
    def __init__(self,):
    # training hyperparameters
        self.help = 0
        self.architecture = k_sequential
    
    def set_architecture(self,preset):
        if preset == 'tt':
            self.architecture = k_transferTest
        if preset == '2d':
            self.architecture = pit_class_2d
        if preset == 'conv3d':
            self.architecture = conv3d
        if preset == 'ktt':
            self.architecture = ktt_functional
        
    
    def create_model(self,metrics=['acc']):
        """
        Create and compile an instance of the model
        """
        model = self.architecture() #k_sequential()
        #model.compile(adam(),loss = 'binary_crossentropy',metrics=['acc'])
        model.compile(adam(),loss = 'binary_crossentropy',metrics=metrics)
        return model


catAcc = keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
cce = keras.losses.CategoricalCrossentropy()

def adam(lr = 0.0002):
    """
    ps. Adam optimiser is unaffected by label weighting techniques
    used when there is a large S/B asymetery 
    """
    return keras.optimizers.Adam(
        lr = lr, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=None, 
        decay=0.0, 
        amsgrad=False)


def create_model(
        network = k_sequential(),
        opt = adam(),
        loss = 'binary_crossentropy',
        metrics=['acc'],
    ):
    model = network
    opti = opt
    model.compile(opti,loss,metrics)
    
    # TODO - Make optional
    print(' Creating new model ')
    model.summary()

    return model






def make_inference_model( path='./MLout/fold-1.h5',
                          layercut=10,shape=(480,640,8),
                          verbose=True):
    """
    - Take existing CNN model and converts to FCNN with reduced output dims.
    - Transfer layers up until 'layer-cut' into a new model with input 'shape'.
    - Assumes no retraining necessary, compiles w generic metrics.
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
        y = [ model.predict(x,verbose=0, steps = 1)[0] for model in self.models ]
        return np.concatenate(y,axis=-1)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
import numpy as np
from tools import display, binary

# dirty_train ->  exposed; dirty_test -> exposed_reverse; clean_train -> clean; clean test -> clean_r

dirty_train = np.load("train_data/dirty.npy", allow_pickle=True)
dirty_test = np.load("train_data/dirty_reverse.npy", allow_pickle=True)
clean_train = np.load("train_data/clean.npy", allow_pickle=True)
clean_test = np.load("train_data/clean_reverse.npy", allow_pickle=True)

clean_train = binary(clean_train, 145)
clean_test = binary(clean_test, 145)

clean_test = clean_test / 255
clean_train = clean_train / 255
dirty_test = dirty_test / 255
dirty_train = dirty_train / 255
clean_train = clean_train[:, :, :].reshape((clean_train.shape[0], clean_train.shape[1], clean_train.shape[2], 1))
clean_test = clean_test[:, :, :].reshape((clean_test.shape[0], clean_test.shape[1], clean_test.shape[2], 1))


clean_train = np.pad(clean_train, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant', constant_values=1)
dirty_train = np.pad(dirty_train, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant', constant_values=1)
clean_test = np.pad(clean_test, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant', constant_values=1)
dirty_test = np.pad(dirty_test, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant', constant_values=1)


def build_model(input_layer, layer_factor):
    conv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    # pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(layer_factor * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(layer_factor * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(layer_factor * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(layer_factor * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(layer_factor * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(layer_factor * 16, (3, 3), activation="relu", padding="same")(convm)

    deconv4 = Conv2DTranspose(layer_factor * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(layer_factor * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(layer_factor * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(layer_factor * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(layer_factor * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(layer_factor * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(layer_factor * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    # uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(layer_factor * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    # uconv1 = concatenate([deconv1, conv1])
    # uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(deconv1)
    uconv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = tf.keras.Model(input_layer, output_layer)

    return model


input_shape = Input((None, None, 10))
unet = build_model(input_shape, 8)
unet.summary()

tf.keras.utils.plot_model(unet, show_shapes=True)

unet.compile(optimizer='adam',
             loss="binary_crossentropy",
             metrics=['accuracy'])

TRAIN_LENGTH = dirty_train.shape[0]
BATCH_SIZE = 4
EPOCHS = 35
STEPS_PER_EPOCH = TRAIN_LENGTH // (BATCH_SIZE * EPOCHS)

train_dataset = tf.data.Dataset.from_tensor_slices((dirty_train[:, :, :, :], clean_train[:, :, :, :]))
validation_dataset = tf.data.Dataset.from_tensor_slices((dirty_test[:, :, :, :], clean_test[:, :, :, :]))
train_dataset = train_dataset.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
validation_dataset = validation_dataset.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)

unet_history = unet.fit(train_dataset, epochs=EPOCHS,
                        validation_data=validation_dataset,
                        callbacks=tf.keras.callbacks.Callback(),
                        verbose=2)

pr = unet.predict(dirty_test)

for j in range(249):
    display([dirty_test[j, :, :, 8], clean_test[j, :, :, 0], pr[j, :, :, 0]])

unet.save("model/unet")
np.save("/content/drive/My Drive/MoEDAL/final/model/unet_predict", pr)
#np.save("/content/drive/My Drive/MoEDAL/final/model/unet_testclean", clean_test)
#np.save("/content/drive/My Drive/MoEDAL/final/model/unet_testdirty", dirty_test)

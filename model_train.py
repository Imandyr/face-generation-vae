# train of correct loss calculating VAE model for face generation


"""imports"""

import os, sys, random, time, glob, re, shutil, io
from zipfile import ZipFile
import numpy as np
import pandas as pd
from logger import logger

import asyncio
import aiofiles

import skimage
import cv2
import imageio
from PIL import Image

from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers, models, Model, Input, Sequential, optimizers, losses, callbacks
from keras import backend as K

from common_functions import *
from data_generator import *
from vae_model import *


"""params of model"""

path_to_images = r"F:\large_data\Flickr_Faces_HQ_dataset\splitted_images"
batch_size = 32
image_array_shape = (128, 128, 3)
augmentation = False

output_encoder_dim = 10
reconstruction_loss_factor = 1.0

learning_rate = 0.000001
steps_per_epoch = 1280
validation_steps = 160
epochs = 1000

# path to directory for callback outputs
path_to_callback_generated_images = r"generated_images/callbacks_train_images_3"

test_images_patience = 1
model_checkpoint_filepath = "models_checkpoints/VAE_v1.2.h5"
model_checkpoint_monitor = "val_loss"
early_stopping_monitor = "val_loss"
early_stopping_patience = 20
RLROP_monitor = "val_loss"
RLROP_factor = 0.1
RLROP_patience = 5
RLROP_min_lr = 0.000000001
tensorboard_logdir = "tensorboard_logdir/VAE_v1"


"""create data generators"""

# images augmentation generator
augmentation_datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest", )

# initialize generators
train_gen = BaseDataGen(path_to_images+r"\train", batch_size, image_array_shape, augmentation, augmentation_datagen)
val_gen = BaseDataGen(path_to_images+r"\val", batch_size, image_array_shape, augmentation, augmentation_datagen)
test_gen = BaseDataGen(path_to_images+r"\test", batch_size, image_array_shape, augmentation, augmentation_datagen)


"""create and compile model"""

# create model
encoder, decoder, vae_model = VAE(sampling_function=sampling, img_shape=image_array_shape,
                                  output_encoder_dim=output_encoder_dim,
                                  reconstruction_loss_factor=reconstruction_loss_factor,).model()
# build model
vae_model.build(input_shape=[None, image_array_shape[0], image_array_shape[1], image_array_shape[2]])
# summary of vae
vae_model.summary()
# summary of encoder
encoder.summary()
# summary of decoder
decoder.summary()

# compile model
vae_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=vae_loss,
            metrics=[])
vae_model.load_weights("models_checkpoints/VAE_v1.1.h5")


"""train model"""

# test images array
test_images_array = next(test_gen.data_generation())[0]["encoder_input"]

# callbacks
callbacks_list = [
    SaveImageExampleCallback(test_images_array=test_images_array,
                             path_to_generated_images=path_to_callback_generated_images, patience=test_images_patience),
    callbacks.ModelCheckpoint(filepath=model_checkpoint_filepath, monitor=model_checkpoint_monitor, save_best_only=True,
                              save_weights_only=True),
    callbacks.EarlyStopping(monitor=early_stopping_monitor, patience=early_stopping_patience),
    callbacks.ReduceLROnPlateau(monitor=RLROP_monitor, factor=RLROP_factor, patience=RLROP_patience, min_lr=RLROP_min_lr),
    callbacks.TensorBoard(log_dir=tensorboard_logdir),
]

# train
history = vae_model.fit(x=train_gen.data_generation(), steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_data=val_gen.data_generation(), validation_steps=validation_steps,
                        shuffle=True, callbacks=callbacks_list)

# train history
# title
plt.suptitle("history of model training")
# set context
sns.set_context(context="notebook", font_scale=1.0)
sns.set_style(style="darkgrid", rc={'grid.color': '.5'})
# plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
# legend
plt.legend(['train_loss', 'validation_loss', 'train_accuracy', 'validation_accuracy'])
# show
plt.show()





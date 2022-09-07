# test of VAE model for face generation


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
from model import *


"""params of model"""

path_to_images = r"F:\large_data\Flickr_Faces_HQ_dataset\splitted_images"
batch_size = 15
image_array_shape = (128, 128, 3)
augmentation = False

output_encoder_dim = 20
reconstruction_loss_factor = 1000

learning_rate = 0.000001
steps_per_epoch = 200
validation_steps = 40
epochs = 1000

# path to directory for callback outputs
path_to_callback_generated_images = r"generated_images/callbacks_train_images_3/selu_5_3_5_20_16384_2048_1"

test_images_patience = 1
model_checkpoint_filepath = "models_checkpoints/VAE_v1.h5"
model_checkpoint_monitor = "val_loss"
early_stopping_monitor = "val_loss"
early_stopping_patience = 100
RLROP_monitor = "val_loss"
RLROP_factor = 0.1
RLROP_patience = 10
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
vae_model = VAE(sampling_function=sampling, img_shape=image_array_shape, output_encoder_dim=output_encoder_dim)
# build model
vae_model.build(input_shape=[None, image_array_shape[0], image_array_shape[1], image_array_shape[2]])
# summary of vae
vae_model.summary()
# summary of encoder
vae_model.get_layer("encoder_model").summary()
# summary of decoder
vae_model.get_layer("decoder_model").summary()

# initialize loss functions for vae
vae_loss_func_v1 = loss_func_v1(vae_model.get_layer("encoder_model").get_layer("encoder_mu"),
                          vae_model.get_layer("encoder_model").get_layer("encoder_log_variance"),
                          reconstruction_loss_factor=reconstruction_loss_factor)
vae_loss_func_v2 = loss_func_v2(vae_model.get_layer("encoder_model").get_layer("encoder_mu"),
                          vae_model.get_layer("encoder_model").get_layer("encoder_log_variance"))

# compile model
vae_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=vae_loss_func_v1,
            metrics=["accuracy"])

# load weights
vae_model.load_weights("models_checkpoints/VAE_v1.h5")


"""test, generate and save images"""

# create txt file for values
f = open(r"generated_images\gen_1\values.txt", mode="w", encoding="utf-8")

# decode and save generated images

for count in range(0, 1000000):
    # generate random values for image generation
    random_values = []
    for count2 in range(0, output_encoder_dim):
        random_values.append(float(random.uniform(-10., 10.)))
    # to numpy array
    random_values = np.asarray(random_values).astype("float32").reshape(1, output_encoder_dim)
    # generate image
    generated_image = vae_model.decoder(random_values)
    # de-standardize
    p_image_array = de_standardization(generated_image[0])
    # write image
    imageio.imwrite(
        uri=rf"generated_images\gen_1\generated_image_{count}.jpeg",
        im=p_image_array)
    # write value
    f.write(str(random_values.tolist())+"\n")




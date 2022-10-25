# test of VAE model for face generation with corrected loss calculation


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

steps_per_epoch = 50

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

vae_model.load_weights("models_checkpoints/VAE_v1.2.h5")


"""test, generate and save images"""

# create txt file for values
f = open(r"generated_images\gen_3\values.txt", mode="w", encoding="utf-8")

# decode and save generated images
for count in range(0, 1000):
    # generate random values for image generation
    random_values = np.random.uniform(low=-3., high=3., size=(1, output_encoder_dim)).astype("float32")
    # generate image
    generated_image = decoder.predict(random_values, verbose=0)
    # de-standardize
    p_image_array = de_standardization(generated_image[0])
    # write image
    imageio.imwrite(
        uri=rf"generated_images\gen_3\generated_image_{count}.jpeg",
        im=p_image_array)
    # write value
    f.write(str(random_values.tolist())+"\n")




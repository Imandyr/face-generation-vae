# all useful functions


"""imports"""

import os, sys, random, time, glob, re, shutil, io
from zipfile import ZipFile

import keras.utils
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

from tensorflow import math
from keras import layers, models, Model, Input, Sequential, optimizers, losses, callbacks, activations, constraints
from keras import backend as K


"""functions"""


# standard standardization
def standardization(image_array):
    image_array = np.asarray(image_array).astype("float32")
    mean, std = image_array.mean(), image_array.std()
    image_array = (image_array - mean) / std
    image_array = np.asarray(image_array).astype("float32")
    return image_array


# fixed standardization for images in range [0, 255]
def fixed_image_standardization(image_array):
    image_array = np.asarray(image_array).astype("float32")
    # standardization by subtracting and dividing image values by mean(127.5)
    image_array = (image_array - 127.5) / 127.5
    image_array = np.asarray(image_array).astype("float32")
    return image_array


# de-standardization for images to uint8 [0, 255]
def de_standardization(raw_image_array):
    raw_image_array = np.asarray(raw_image_array)
    # decrease image values, so that they become in the area [-1, 1]
    image_array = (raw_image_array - np.min(raw_image_array)) / (np.max(raw_image_array) - np.min(raw_image_array))
    # crop values in [-1, 1] to avoid errors
    image_array = np.clip(image_array, -1, 1)
    # conversion image array to uint8 using skimage.img_as_ubyte()
    image_array = skimage.util.img_as_ubyte(image_array)
    image_array = np.asarray(image_array).astype("uint8")
    return image_array


# weights centralization function
class WeightCentralization(constraints.Constraint):
    def __call__(self, w):
        return w - math.reduce_mean(w)


# callback for saving images during model training
class SaveImageExampleCallback(callbacks.Callback):
    """
        callback for saving image while train process
    """
    # initialize
    def __init__(self, test_images_array, path_to_generated_images: str, patience=0):
        super().__init__()

        # array with images to test model
        self.test_images_array = test_images_array
        # path to directory for generated images
        self.path_to_generated_images = path_to_generated_images
        # how many epochs been skipped between one image generation test
        self.patience = patience

        # count of epoch
        self.count_of_epoch = 0
        # count of skipped epochs
        self.count_of_skipped_epoch = 0

        # save original test images for easier comparison
        self.save_original()

    # save original test images to target directory for easier comparison
    def save_original(self):
        for c1, image_array in enumerate(self.test_images_array):
            p_image_array = de_standardization(image_array)
            imageio.imwrite(
                uri=rf"{self.path_to_generated_images}/original_test_image_{c1}.jpeg",
                im=p_image_array)

    # what's going on in epoch end
    def on_epoch_end(self, epoch, logs=None):
        # count epoch
        self.count_of_epoch += 1
        self.count_of_skipped_epoch += 1

        # if count of skipped epochs == patience, process images in model and save
        if self.count_of_skipped_epoch == self.patience:
            # reset skipped epoch counter
            self.count_of_skipped_epoch = 0

            # prepare model to image generation
            generative_model = keras.models.Model(
                self.model.get_layer(name="encoder_input").input, self.model.get_layer(name="loss_layer").input[1])

            # generate images
            generated_images = generative_model.predict(self.test_images_array)

            # decode and save images to generated images directory
            for c1, image_array in enumerate(generated_images):
                p_image_array = de_standardization(image_array)
                imageio.imwrite(
                    uri=rf"{self.path_to_generated_images}/epoch_{self.count_of_epoch}_image_{c1}_.jpeg",
                                im=p_image_array)
















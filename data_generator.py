# data generator

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

from common_functions import *


"""create base data generator"""


# define base generator class
class BaseDataGen:
    # define init
    def __init__(self, path_to_images: str, batch_size: int, image_array_shape: tuple, augmentation: bool, augmentation_datagen=None):
        # path to folder with target images
        self.path_to_images = path_to_images
        # size of output images arrays
        self.batch_size = batch_size
        # shape of image
        self.img_h = image_array_shape[0]
        self.img_w = image_array_shape[1]
        self.img_c = image_array_shape[2]
        # augmentation
        self.augmentation = augmentation
        # augmentation generator
        self.augmentation_datagen = augmentation_datagen

    # single image processing
    def image_processing(self, image):
        image_array = utils.load_img(image, target_size=(self.img_h, self.img_w))
        image_array = utils.img_to_array(image_array)
        image_array = np.asarray(image_array).astype("float32").reshape(self.img_h, self.img_w, self.img_c)
        return image_array

    # image generator function
    def data_generation(self):
        # iterate data generator loop for eternity
        while True:
            # load all images names from path_to_images and shuffle them
            images_names = np.asarray(os.listdir(self.path_to_images)).astype("str")
            np.random.shuffle(images_names)
            # create array for batch of processed images
            images_batch = []

            # iterate load and process all images from images_names
            for c1, image_name in enumerate(images_names):
                # load and process image
                image_array = self.image_processing(f"{self.path_to_images}/{image_name}")
                # add to batch
                images_batch.append(image_array)

                # when images_batch accumulate batch of image, it yields this batch
                if len(images_batch) >= self.batch_size:
                    # list to array
                    images_batch = np.asarray(images_batch).astype("float32").reshape(self.batch_size, self.img_h,
                                                                                      self.img_w, self.img_c)
                    # apply augmentation, if augmentation == True
                    if self.augmentation:
                        images_batch = next(self.augmentation_datagen.flow(images_batch, None,
                                                                           batch_size=self.batch_size, shuffle=False))

                    # standardize images batch
                    images_batch = fixed_image_standardization(images_batch)

                    # yield images batch
                    yield {'encoder_input': images_batch}, {"loss_layer": images_batch}
                    # reset images_batch variable
                    images_batch = []






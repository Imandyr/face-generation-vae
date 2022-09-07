# all useful functions


"""imports"""

import os, sys, random
import numpy as np
import pandas as pd
from logger import logger

import skimage
import cv2
import imageio
from PIL import Image

from keras import utils
from sklearn import preprocessing as pp

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










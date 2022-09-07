# split all image data to train-validation-test datasets and save to directories


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
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


"""create functions for load, split, process and save data"""


# load zip archive with images
def load_zip(zip_path: str):
    archive = ZipFile(zip_path, 'r')
    logger.info("archive been loaded")
    return archive


# load and split all names
# archive: loaded zip archive ; train_size: percent of data in train set (0.80 = 80%)
def load_and_split_filenames(archive, train_size: float):
    # take names of all images
    images_names = archive.namelist()

    # split images names to train, validation and test datasets
    x_train, x_test = train_test_split(images_names, train_size=train_size, shuffle=True)
    x_val, x_test = train_test_split(x_test, train_size=0.5, shuffle=True)

    # return names
    logger.info("names been splitted")
    return x_train, x_val, x_test


# function for asynchronous way to save image
async def image_save(image, path_to_image: str):
    async with aiofiles.open(path_to_image, "wb") as file:
        await file.write(image)


# function for processing of single image
async def load_and_save_image(archive, base_path_to_splits: str, split_name: str, image_name: str, img_h: int, img_w: int):
    # load and process image
    loaded_image = archive.read(image_name)
    loaded_image = io.BytesIO(loaded_image)
    loaded_image = utils.load_img(loaded_image, target_size=(img_h, img_w))
    # create buffer for image
    buffer = io.BytesIO()
    # load image to buffer as JPEG
    loaded_image.save(buffer, format="JPEG")
    # asynchronously save image from to dataset
    await image_save(buffer.getbuffer(), rf"{base_path_to_splits}\{split_name}\{image_name[:-4]}.jpeg")


# function for create tasks to processing for all images
async def load_and_save_images(archive, base_path_to_splits: str, split_name: str, images_names: list, img_h: int, img_w: int):
    # list for tasks
    task_list = []
    # iterate and create tasks for names
    for image_name in images_names:
        # create task for image save and processing
        task = await asyncio.create_task(load_and_save_image(archive, base_path_to_splits, split_name, image_name, img_h, img_w))
        # append to task list
        task_list.append(task)
    # return task list
    return task_list


# process and save images to datasets
async def process_and_save_images_to_splits(archive, train_names: list, val_names: list, test_names: list,
                                      base_path_to_splits: str, img_h: int, img_w: int):

    # create directories for datasets
    try:
        # try to delete previous directories with files
        shutil.rmtree(rf"{base_path_to_splits}\train")
        shutil.rmtree(rf"{base_path_to_splits}\val")
        shutil.rmtree(rf"{base_path_to_splits}\test")
        time.sleep(5)
        logger.info("previous images in directories been deleted")
    except:
        pass
    finally:
        # create new directories
        os.makedirs(rf"{base_path_to_splits}\train")
        os.makedirs(rf"{base_path_to_splits}\val")
        os.makedirs(rf"{base_path_to_splits}\test")
        logger.info("directories for datasets been created")

    # save images

    # test
    logger.info("start of test images writing")
    # create tasks
    task_list = await load_and_save_images(archive, base_path_to_splits, "test", test_names, img_h, img_w)
    # execution all task
    asyncio.wait(task_list)
    logger.info("end of test images writing")

    # val
    logger.info("start of val images writing")
    # create tasks
    task_list = await load_and_save_images(archive, base_path_to_splits, "val", val_names, img_h, img_w)
    # execution all task
    asyncio.wait(task_list)
    logger.info("end of val images writing")

    # train
    logger.info("start of train images writing")
    # create tasks
    task_list = await load_and_save_images(archive, base_path_to_splits, "train", train_names, img_h, img_w)
    # execution all task
    asyncio.wait(task_list)
    logger.info("end of train images writing")


"""use all functions"""

# params
base_images_path = r"F:\large_data\Flickr_Faces_HQ_dataset\original_array\archive.zip"
base_splitted_images_path = r"F:\large_data\Flickr_Faces_HQ_dataset\splitted_images"
train_size = 0.80
img_h, img_w = 512, 512

# load archive
archive = load_zip(zip_path=base_images_path)

# load and split images names
x_train, x_val, x_test = load_and_split_filenames(archive=archive, train_size=train_size)

# save images to split
asyncio.run(process_and_save_images_to_splits(archive=archive, train_names=x_train, val_names=x_val, test_names=x_test,
                                  base_path_to_splits=base_splitted_images_path, img_h=img_h, img_w=img_w))












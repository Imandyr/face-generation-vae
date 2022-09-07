# create VAE model for face generation


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

from tensorflow import math
from keras import layers, models, Model, Input, Sequential, optimizers, losses, callbacks, activations, constraints
from keras import backend as K

from common_functions import *


"""define model"""


# callback for saving images during model training
class SaveImageExampleCallback(callbacks.Callback):
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
        print(epoch)
        # count epoch
        self.count_of_epoch += 1
        self.count_of_skipped_epoch += 1

        # if count of skipped epochs == patience, process images in model and save
        if self.count_of_skipped_epoch == self.patience:
            # reset skipped epoch counter
            self.count_of_skipped_epoch = 0

            # take params
            lr = round(float(K.get_value(self.model.optimizer.learning_rate)), 8)
            loss = round(float(logs.get('val_loss')), 5)
            accuracy = round(float(logs.get('val_accuracy')), 4)

            # generate images
            generated_images = self.model.predict(self.test_images_array)

            # decode and save images to generated images directory
            for c1, image_array in enumerate(generated_images):
                p_image_array = de_standardization(image_array)
                imageio.imwrite(
                    uri=rf"{self.path_to_generated_images}/epoch_{self.count_of_epoch}_image_{c1}_lr_{lr}_loss_{loss}_acc_{accuracy}_.jpeg",
                                im=p_image_array)


# function for calculate random sampling of mean and variance
def sampling(mu_log_variance):
    # take mean and variance
    mu, log_variance = mu_log_variance
    # calculate random normal distribution of variance
    epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
    # calculate random sampling
    random_sample = mu + K.exp(log_variance / 2) * epsilon
    # return
    return random_sample


# vae loss function v1
def loss_func_v1(encoder_mu, encoder_log_variance, reconstruction_loss_factor: int):

    # calculate reconstruction loss (compare of true and generated image)
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss = K.mean(K.square(y_true-y_predict))
        return reconstruction_loss_factor * reconstruction_loss

    # calculate loss in encoder mean and variance generation
    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
        return kl_loss

    # calculate metric of loss in encoder mean and variance generation?
    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
        return kl_loss

    # calculate vae from summary of reconstruction_loss and kl_loss
    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss + kl_loss
        return loss

    # return final loss value
    return vae_loss


# vae loss function v2
def loss_func_v2(encoder_mu, encoder_log_variance):
    # calculate reconstruction loss (compare of true and generated image)
    def vae_reconstruction_loss(y_true, y_predict):
        return K.mean(K.square(y_true - y_predict))

    # calculate loss in encoder mean and variance generation
    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * K.mean(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance),
                                axis=-1)
        return kl_loss

    # calculate vae from summary of reconstruction_loss and kl_loss
    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss + 0.03 * kl_loss
        return loss

    # return final loss value
    return vae_loss


# VAE model
class VAE(Model):
    # initialize all variables
    def __init__(self, sampling_function, img_shape: tuple, output_encoder_dim: int):
        super().__init__()
        # function for calculate sampling in encoder output
        self.sampling_function = sampling_function
        # original shape of images
        self.img_h, self.img_w, self.img_c = img_shape[0], img_shape[1], img_shape[2]
        # number of dimensions in encoder output
        self.output_encoder_dim = output_encoder_dim
        # shape of data in encoder before flattering
        self.shape_before_flatten = []
        # build encoder models
        self.encoder = self.encoder_model()
        # build decoder model
        self.decoder = self.decoder_model()

    # define encoder
    def encoder_model(self):
        encoder_conv_kernel_size = (3, 3)
        # encoder convolution block
        def encoder_conv_block(x, filters, kernel_size, padding, strides, number_of_layer):
            x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides,
                              kernel_initializer='lecun_normal',
                              name=f"encoder_conv_{number_of_layer}")(x)
            x = layers.BatchNormalization(name=f"encoder_norm_{number_of_layer}")(x)
            x = layers.Activation(activation=activations.selu, name=f"encoder_activation_{number_of_layer}")(x)
            return x
        # input
        encoder_input = layers.Input(shape=(self.img_h, self.img_w, self.img_c), dtype="float32", name="encoder_input")
        # convolution
        x = encoder_conv_block(encoder_input, filters=32, kernel_size=(3, 3), padding="same", strides=1,
                               number_of_layer=1)
        x = encoder_conv_block(x, filters=64, kernel_size=encoder_conv_kernel_size, padding="same", strides=2, number_of_layer=2)
        x = encoder_conv_block(x, filters=128, kernel_size=encoder_conv_kernel_size, padding="same", strides=2, number_of_layer=3)
        x = encoder_conv_block(x, filters=256, kernel_size=encoder_conv_kernel_size, padding="same", strides=2, number_of_layer=4)
        x = encoder_conv_block(x, filters=512, kernel_size=encoder_conv_kernel_size, padding="same", strides=2, number_of_layer=5)
        x = encoder_conv_block(x, filters=1024, kernel_size=encoder_conv_kernel_size, padding="same", strides=2, number_of_layer=6)
        # flattering
        self.shape_before_flatten = K.int_shape(x)[1:]
        encoder_flatten = layers.Flatten()(x)
        # final dense processing
        x = layers.Dense(2048, kernel_initializer='lecun_normal')(encoder_flatten)  # , kernel_initializer='lecun_normal'
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activations.selu)(x)
        # shape of mean, variance and output vector
        # calculate mean of data
        encoder_mu = layers.Dense(units=self.output_encoder_dim, name="encoder_mu")(x)
        # calculate variance of data
        encoder_log_variance = layers.Dense(units=self.output_encoder_dim, name="encoder_log_variance")(encoder_flatten)
        # calculate and return random sampling of encoder data
        encoder_output = layers.Lambda(function=self.sampling_function, name="encoder_output")([encoder_mu, encoder_log_variance])
        # build encoder model
        encoder = Model(encoder_input, encoder_output, name="encoder_model")
        # return model
        return encoder

    # define decoder
    def decoder_model(self):
        decoder_conv_kernel_size = (5, 5)
        # decoder transpose convolution block
        def decoder_conv_block(x, filters, kernel_size, padding, strides, number_of_layer):
            x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides,
                              kernel_initializer='lecun_normal',
                              name=f"decoder_conv_tran_{number_of_layer}")(x)
            x = layers.BatchNormalization(name=f"decoder_norm_{number_of_layer}")(x)
            x = layers.Activation(activation=activations.selu, name=f"decoder_activation_{number_of_layer}")(x)
            return x
        # input
        decoder_input = layers.Input(shape=(self.output_encoder_dim,), dtype="float32", name="decoder_input")
        # first dense processing
        x = layers.Dense(2048, kernel_initializer='lecun_normal')(decoder_input)  # , kernel_initializer='lecun_normal'
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activations.selu)(x)
        # expand size of vector
        x = layers.Dense(units=np.prod(self.shape_before_flatten), name="decoder_dense_2")(x)
        # reshape vector
        x = layers.Reshape(target_shape=self.shape_before_flatten, name="reshape_1")(x)
        # reconstruction
        x = decoder_conv_block(x, filters=512, kernel_size=decoder_conv_kernel_size, padding="same", strides=2, number_of_layer=1)
        x = decoder_conv_block(x, filters=256, kernel_size=decoder_conv_kernel_size, padding="same", strides=2, number_of_layer=2)
        x = decoder_conv_block(x, filters=128, kernel_size=decoder_conv_kernel_size, padding="same", strides=2, number_of_layer=3)
        x = decoder_conv_block(x, filters=64, kernel_size=decoder_conv_kernel_size, padding="same", strides=2, number_of_layer=4)
        x = decoder_conv_block(x, filters=32, kernel_size=decoder_conv_kernel_size, padding="same", strides=2, number_of_layer=5)
        # final transpose to original size and send to output
        x = layers.Conv2DTranspose(filters=3, kernel_size=decoder_conv_kernel_size, padding="same", strides=1,
                                   name=f"decoder_conv_tran_6")(x)
        x = layers.BatchNormalization(name=f"decoder_norm_6")(x)
        decoder_output = layers.Activation(activation=activations.sigmoid, name=f"decoder_output")(x)
        # build model
        decoder = Model(decoder_input, decoder_output, name="decoder_model")
        # return model
        return decoder

    # call VAE
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



import argparse
import os

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Input, Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Input, average)
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

import videoto3d


def create_3dcnn(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3,3,3), input_shape=(
        input_shape), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3,3,3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3,3,3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3,3,3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

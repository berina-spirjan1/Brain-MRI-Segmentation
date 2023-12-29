import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.preprocessing.image import ImageDataGenerator
plt.style.use('ggplot')


def plot_from_image_path(number_of_rows, number_of_columns, list_of_image_paths, list_of_mask_paths):
    fig = plt.figure(figsize=(12, 12))
    # we are building a grid here, that's the reason why we need rows*columns
    for i in range(1, number_of_rows * number_of_columns + 1):
        fig.add_subplot(number_of_rows, number_of_columns, i)
        image_path = list_of_image_paths[i]
        mask_path = list_of_mask_paths[i]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
    plt.show()


def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)


def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou


def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)

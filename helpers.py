import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.style.use('ggplot')

def plot_from_image_path(number_of_rows, number_of_columns, list_of_image_paths,list_of_mask_paths):
    fig = plt.figure(figsize=(12,12))
    #we are building a grid here, that's the reason why we need rows*columns
    for i in range(1,number_of_rows*number_of_columns+1):
        fig.add_subplot(number_of_rows,number_of_columns,i)
        image_path = list_of_image_paths[i]
        mask_path = list_of_mask_paths[i]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
    plt.show()
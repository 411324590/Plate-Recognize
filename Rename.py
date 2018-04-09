# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
import cv2


def Rename(path1):
    x_img = []
    y_name = []
    filelist = [os.path.join(path1, f) for f in os.listdir(path1)]
    Img_name = np.array(os.listdir(path1))
    Img_name = [x[0:13] for x in Img_name]

    #print(y_test)
    return x_img,y_name
# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
import cv2

from  gencarplate import GenPlate
G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]

chars_Index = {0:"京",1: "沪",2: "津", 3:"渝", 4:"冀", 5:"晋",6:"蒙", 7:"辽", 8:"吉", 9:"黑", 10:"苏", 11:"浙", 12:"皖", 13:"闽", 14:"赣",
         15:"鲁", 16:"豫", 17:"鄂", 18:"湘", 19:"粤", 20:"桂",21:"琼", 22:"川",23: "贵",24: "云",25: "藏",26: "陕",27: "甘", 28:"青",
         29:"宁", 30:"新",31: "0", 32:"1",33: "2",34: "3",35: "4", 36:"5", 37:"6", 38:"7",39: "8", 40:"9",41: "A",42:"B",43: "C",
         44:"D",45: "E", 46:"F", 47:"G", 48:"H",49: "J", 50:"K", 51:"L", 52:"M",53: "N",54: "P",55: "Q",56: "R", 57:"S",58: "T",
         59:"U",60: "V", 61:"W",62: "X",63:"Y", 64:"Z" }

def load_data(path1):
    x_img = []
    y_name = []
    filelist = [os.path.join(path1, f) for f in os.listdir(path1)]
    Img_name = np.array(os.listdir(path1))
    Img_name = [x[6:13] for x in Img_name]
    n1 = len(filelist)
    for img in filelist:
        im = np.array(Image.open(img).resize((272, 72)),dtype=np.uint8)
        x_img.append(im)
    x_img = np.array(x_img)
    #print(x_test)
    ytmp = np.array(list(map(lambda x: [index[a] for a in list(x)], Img_name)), dtype=np.uint8)
    #print(ytmp)
    y = np.zeros([ytmp.shape[1], n1, len(chars)])
    for batch in range(n1):
        for idx, row_i in enumerate(ytmp[batch]):
            y[idx, batch, row_i] = 1
    #print(y)
    y_name = [yy for yy in y]
    #print(y_test)
    return x_img,y_name

def load_Img(path1):
    x_img = []
    filelist = [os.path.join(path1, f) for f in os.listdir(path1)]
    Img_name = np.array(os.listdir(path1))
    Img_name = [x[6:13] for x in Img_name]
    n1 = len(filelist)
    for img in filelist:
        im = np.array(Image.open(img).resize((272, 72)), dtype=np.uint8)
        x_img.append(im)
    x_img = np.array(x_img)
    return x_img


# x_test =load_Img("testImg")
# print(x_test.shape)

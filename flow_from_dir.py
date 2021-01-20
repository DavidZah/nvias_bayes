import glob
import os
import numpy as np
import cv2
from os import walk



def load_as_vec(lst):
    thresh = 128
    category_lst = []
    for i in lst:
        f = []
        vec_lst = []
        for (dirpath, dirnames, filenames) in walk(i):
            f.extend(filenames)

        for j in f:
            load_pic = cv2.imread(i+'//'+j, cv2.IMREAD_GRAYSCALE)
            img_bin = np.array(cv2.threshold(load_pic, thresh, 255, cv2.THRESH_BINARY)[1]).ravel()
            vec_lst.append(img_bin)
        category_lst.append(vec_lst)
    print(category_lst)

def scan_dir(path):
    list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
    return list_subfolders_with_paths

if __name__ == "__main__":
    dirs = scan_dir('numbers//')
    load_as_vec(dirs)
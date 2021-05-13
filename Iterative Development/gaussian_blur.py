import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

i = 0
os.chdir("F:/Uni/Software Engineering Practice/Iterative Development/dataset/images")
for file in glob.glob("*.png"):
    os.chdir("F:/Uni/Software Engineering Practice/Iterative Development/dataset/images")
    print(i)
    i += 1
    img = cv2.imread(file)
    os.chdir("F:/Uni/Software Engineering Practice/Iterative Development/dataset/blurred")
    blur = cv2.GaussianBlur(img,(5,5),0)
    filename = file
    cv2.imwrite(filename,blur)

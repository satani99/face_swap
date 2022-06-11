# python content.py --source Path to source --destination Path to destination --out Output Path


import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--source', required=True, help='Path for source image')
    parser.add_argument('--destination', required=True, help='Path for target image')
    parser.add_argument('--out', required=True, help='Path for storing output images')
    args = parser.parse_args()


    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(args.destination)
    source = cv2.imread(args.source)
    img1 = Image.open(args.destination)
    img2 = Image.open(args.source)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    x, y, w, h = faces[0]
    x_inc = int(w*0.50)
    y_inc = int(h*0.50)
    img2 = img2.resize((w+x_inc, h+y_inc))
    img1.paste(img2, (x-int(w*0.25), y-int(h*0.3)), mask=img2)


    img1.save(args.out)
    
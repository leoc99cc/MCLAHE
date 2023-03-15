import mclahe as mc
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import cv2
from PIL import Image

img_name = '10.jpeg'
img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2)
clahe_img = clahe.apply(img)
cv2.imwrite('10clahe_img.jpg', clahe_img)
cv2.imwrite('10gray_img.jpg', img)

class Run:
    img_name = 'DR1_Local_t000_ch00_mixSmall.jpg'
    contrast = -100

    def __init__(self):
        self.down_contrast()
        self.malache()

    def down_contrast(self):
        img = cv2.imread(self.img_name, flags=1)
        brightness = 0
        B = brightness / 255.0
        c = self.contrast / 255.0
        k = math.tan((45 + 44 * c) / 180 * math.pi)
        img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(re.sub('.jpg', '', self.img_name) + '_dc_' + str(self.contrast) + '.jpg', img)


    def malache(self):
        img = plt.imread(re.sub('.jpg', '', self.img_name) + '_dc.jpg')
        img_mclahe = mc.mclahe(img[:, :, 0])
        cv2.imwrite(re.sub('.jpg', '', self.img_name) + '_mclahe_' + str(self.contrast) + '.jpg', img_mclahe)






import numpy as np
import imgaug.augmenters as iaa
import cv2
from os import listdir
from os.path import isfile, join

imageDir = "./images/"
saveDir = "./augmentedImages/"

imageFiles = [f for f in listdir(imageDir) if isfile(join(imageDir, f))]



def load_batch(batch_idx):
    # dummy function, implement this
    # Return a numpy array of shape (N, height, width, #channels)
    # or a list of (height, width, #channels) arrays (may have different image
    # sizes).
    # Images should be in RGB for colorspace augmentations.
    # (cv2.imread() returns BGR!)
    # Images should usually be in uint8 with values from 0-255.

    images = []
    for i in imageFiles:
        a = cv2.imread(imageDir + i)
        #a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        images.append(a)
    return images

def train_on_images(images):
    # dummy function, implement this
    pass

# Pipeline:
# (1) Crop images from each side by 1-16px, do not resize the results
#     images back to the input size. Keep them at the cropped size.
# (2) Horizontally flip 50% of the images.
# (3) Blur images using a gaussian kernel with sigma between 0.0 and 3.0.
seq = iaa.Sequential([
    iaa.Crop(px=(1, 16), keep_size=False),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
])

for batch_idx in range(100):
    otherIndex = 0
    images = load_batch(batch_idx)
    images_aug = seq(images=images)  # done by the library
    for image in images_aug:
        print(".")
        cv2.imwrite("./image_" + str(batch_idx) + str(otherIndex) + ".bmp", image)
        otherIndex+=1

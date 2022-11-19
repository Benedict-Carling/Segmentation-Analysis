"""
Benchmark mask performance against ground truth
"""
import pip  # package installer for python
import numpy as np
from PIL import Image  # PIL(=imaging library), Image(=represent images)
from numpy import asarray

# region Functions
def iou_coef(y_true, y_pred, smooth=1):
    """Intersection-over-Union / Jaccard coefficient"""
    intersection = np.sum(np.abs(y_true * y_pred))  # overlap
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = np.mean((intersection + smooth) / (union + smooth))
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient"""
    intersection = np.sum(y_true * y_pred)
    unionish = np.sum(y_true) + np.sum(y_pred)  # sum of area of target class
    dice = np.mean((2.0 * intersection + smooth) / (unionish + smooth))
    return dice


# endregion

# region Load Images
# im_name = "im128.jpg"
im_truth_name = "images/ground-truth.gif"
im_pred_name = "images/predicted-mask.png"

# Load the normal image
# im = Image.open(im_name)
# im.show() # display image

# Load the ground truth mask
im_truth = Image.open(im_truth_name)
# im_truth.show()

# Load the predicted mask
im_pred = Image.open(im_pred_name)
# im_pred.show()
# endregion

# region Index Calculations
array_truth = asarray(im_truth)
array_pred = asarray(im_pred)

array_pred = array_pred / np.max(array_pred)  # normalising

iouu = iou_coef(array_truth, array_pred, 1e-6)
dicee = dice_coef(array_truth, array_pred, 1e-6)

iou_test = iou_coef(array_truth, array_truth, 1e-6)
dice_test = dice_coef(array_truth, array_truth, 1e-6)

print(iouu)
print(dicee)

print(iou_test)
print(dice_test)

# assert config.IMAGE_MIN_DIM == 256
import glob
import os
import sys

import random
import math
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import *
import pandas as pd
import warnings
import glob
import pickle
ORIG_SIZE = 1024

def predict(model, image_fps, filepath='submission.csv', min_conf=0.95,
            do_trick=False, shape=256):
    # assume square image
    resize_factor = ORIG_SIZE
    # resize_factor = ORIG_SIZE
    with open(filepath, 'w') as file:
        for image_id in tqdm_notebook(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)

            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            if do_trick:
                ratio = oof_lookup[patient_id]
                r['scores2'] = r['scores'] / ratio
            else:
                r['scores2'] = r['scores']

            out_str = stringify_preds(r, patient_id, min_conf, resize_factor)

            file.write(out_str + "\n")
    output = pd.read_csv(filepath, names=['patientId', 'PredictionString'])
    output.to_csv(filepath, index=False)
    return filepath
def make_sub_from_detections(te_dets, filepath='first_ensemble.csv', shape=256):
    # assume square image
    resize_factor = ORIG_SIZE / shape
    # resize_factor = ORIG_SIZE
    with open(filepath, 'w') as file:
        for k,v in tqdm_notebook(te_dets.items(), total=1000):
            patient_id = os.path.splitext(os.path.basename(k))[0]
            print(patient_id)
            out_str = stringify_preds(v, patient_id, 0., resize_factor=resize_factor)
            file.write(out_str + "\n")
    output = pd.read_csv(filepath, names=['patientId', 'PredictionString'])
    output.to_csv(filepath, index=False)
    return filepath

def stringify_preds(r, patient_id, min_conf, resize_factor=1024/256.):
    """Also converts x2, y2 to w,h and resizes"""
    out_str = ""
    out_str += patient_id
    out_str += ","
    assert (len(r['rois']) == len(r['class_ids']) == len(r['scores']))
    if len(r['rois']) == 0:
        pass
    else:
        num_instances = len(r['rois'])
        for i in range(num_instances):
            score = r['scores'][i]
            if score > min_conf:
                out_str += ' '
                out_str += str(round(r['scores'][i], 2))
                out_str += ' '

                # x1, y1, width, height
                x1 = r['rois'][i][1]
                y1 = r['rois'][i][0]
                width = r['rois'][i][3] - x1
                height = r['rois'][i][2] - y1
                bboxes_str = "{} {} {} {}".format(x1 * resize_factor, y1 * resize_factor, \
                                                  width * resize_factor,
                                                  height * resize_factor)
                out_str += bboxes_str
    return out_str

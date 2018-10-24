import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from tqdm import *
from pathlib import Path

#path = np.random.choice(images)
import funcy
#@funcy.memoize()
prediction_model = None  # global in notebook usually
DATA_DIR = Path('/home/paperspace/data/')
img_dir = DATA_DIR/'images'
test_dcm_dir = DATA_DIR / 'stage_1_test_images'
def get_test_paths():
    test_dcm_fps = list(set(glob.glob(os.path.join(test_dcm_dir, '*.dcm'))))
    test_patient_ids = pd.Series(test_dcm_fps).apply(
        lambda dcm_fp: dcm_fp.strip().split("/")[-1].replace(".dcm", ""))
    test_paths = test_patient_ids.apply(lambda x: os.path.join(img_dir, x + '.jpg'))
    return test_paths
test_paths = get_test_paths()
import glob
import pandas as pd



def run_inference(path):
    image = read_image_bgr(path)
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)

    image, scale = resize_image(image, min_side=val_gen.image_min_side, max_side=val_gen.image_max_side)
    # process image

    start = time.time()
    boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /=scale
    return boxes[0], scores[0], labels[0]

THRESH = 0.0
def get_submit_line_retinanet(image_path, max_detections =2, thresh=THRESH):
    boxes, scores, labels = run_inference(image_path)
    if scores.max() < thresh:
        return ''
    indices = np.where(scores > thresh)[0][:max_detections]
    scores_sort = np.argsort(-scores)[:max_detections]
    assert scores_sort[0] == 0, (scores_sort, scores)
    image_boxes = boxes[indices]
    image_scores = scores[indices]
    image_labes = labels[indices]
    assert image_labes.max() == 0
    submit_line = ''
    for (score, (x,y,x2,y2)) in zip(image_scores, image_boxes):
        submit_line += "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} ".format(score, x, y, x2-x, y2-y)
    return submit_line




def path_to_patient_id(image_path,ext='.jpg'):
    return os.path.basename(image_path).rstrip(ext)


def get_test_paths():
    test_dcm_fps = list(set(glob.glob(os.path.join(test_dcm_dir, '*.dcm'))))
    test_patient_ids = pd.Series(test_dcm_fps).apply(
        lambda dcm_fp: dcm_fp.strip().split("/")[-1].replace(".dcm", ""))
    test_paths = test_patient_ids.apply(lambda x: os.path.join(img_dir, x + '.jpg'))
    return test_paths

def make_submission(sub_path):
    submit_dict = {"patientId": [], "PredictionString": []}
    for image_path in tqdm_notebook(test_paths):
        patient_id = path_to_patient_id(image_path)
        submit_line = get_submit_line_retinanet(image_path, thresh=0.)
        submit_dict["patientId"].append(patient_id)
        submit_dict["PredictionString"].append(submit_line)

    sub = pd.DataFrame(submit_dict).sort_index(axis=1, ascending=False)
    sub.sort_index(axis=1, ascending=False).to_csv(sub_path, index=False)




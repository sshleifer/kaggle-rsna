import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import *
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob

ORIG_SIZE = 1024
DATA_DIR = '/home/paperspace/data'
ROOT_DIR = '/home/paperspace/mask_rcnn_logs'
train_dicom_dir = os.path.join(DATA_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_1_test_images')

# Import Mask RCNN
os.chdir('Mask_RCNN')
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from pathlib import Path


#COCO_WEIGHTS_PATH = "../mask_rcnn_pneumonia_0016.h5"
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations

augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = IMAGE_MIN_DIM
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 200


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')

        # add images
        for i, path in enumerate(image_fps):
            annotations = image_annotations[path]
            self.add_image('pneumonia', image_id=i, path=path,
                           annotations=annotations,
                           orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

coco_weight_path = '/home/paperspace/Mask_RCNN/mask_rcnn_coco.h5'
def train_from_coco(dataset_train, dataset_val, coco_weight_path=coco_weight_path):
    assert Path(coco_weight_path).exists()
    # MAKE SURE YOU ARE ONLY EXCLUDING THE PRETRAINED WEIGHTS THAT YOU WANT
    EXCLUDE_GROUPS = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)
    print('Model Dir: {}'.format(model.model_dir))
    model.load_weights(coco_weight_path, by_name=True,
                       exclude=EXCLUDE_GROUPS)

    LEARNING_RATE = 0.005
    # train heads with higher lr to speedup the learning
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE*2,
                epochs=2,
                layers='heads',
                augmentation=None)  ## no need to augment yet

    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE,
                epochs=6,
                layers='all',
                augmentation=augmentation)

    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE/ 5,
                epochs=16,
                layers='all',
                augmentation=augmentation)


config = DetectorConfig()
anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)
image_fps_list = list(image_fps)
random.seed(42)
random.shuffle(image_fps_list)
N_SPLITS = 3
val_size = 1500
from sklearn.model_selection import KFold
kf = KFold(n_splits=N_SPLITS, shuffle=True)
for i, (train_fp, val_fp) in tqdm(enumerate(kf.split(image_fps)), total=N_SPLITS):
    image_fps_train = image_fps_list[train_fp]
    dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_train.prepare()

    image_fps_val = image_fps_list[val_fp]
    dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_val.prepare()
    model = load_model_for_inference(paths[i])
    APs = compute_batch_ap(model, dataset_val, inference_config, dataset_val.image_ids,
                           do_trick=False)

    train_from_coco(dataset_train, dataset_val)


def find_last_model_foreach_dir(model_dir='/home/paperspace/mask_rcnn_logs/'):
    """"""
    dir_names = next(os.walk(model_dir))[1]
    key = config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)

    if not dir_names:
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find any matching directories under {}".format(model_dir))

    paths = []
    # Pick last directory
    for d in dir_names:
        dir_name = os.path.join(model_dir, d)
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            print('No weight files in {}'.format(dir_name))
        else:
            checkpoint = os.path.join(dir_name, checkpoints[-1])
            paths.append(checkpoint)
    return paths


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
def load_model_for_inference(model_path):
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# import the necessary packages
import numpy as np


def to_str(boxes, score=.95):
    base = "{} {} {} {} {}"
    strs = []
    for (x1, y1, x2, y2) in boxes:
        w = x2 - x1
        h = y2 - y1
        strs.append(base.format(score, x1, y1, w, h))
    return ' '.join(strs)


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # print("overlap", overlap, w, h)
        # delete all indexes from the index list that have where overlap > overlapThresh
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

def run_nms(catted, nms_tresh=0.7):
    new_boxes = {}

    for k, v in catted.iterrows():
        box_lst = []
        for val in v.dropna().values:
            detections = chunks(val.split(), 5)
            for detection in detections:
                score = detection[1:]
                box_lst.append([float(x) for x in detection[1:]])
        if len(box_lst) > 0:
            new_boxes[k] = non_max_suppression_fast(np.array(box_lst), nms_tresh)
        else:
            new_boxes[k] = np.nan
    return new_boxes


def make3_subs(paths):
    pred_files = []
    for i, model_path in enumerate(paths):
        submission_fp = 'sub_mask_rcnn_oof_{}.csv'.format(i)
        model = load_model_for_inference(model_path)
        pred_file = predict(model, test_image_fps, filepath=submission_fp, do_trick=False)
        pred_files.append(pred_file)
    return pred_files



preds_cache = {}
from mrcnn import utils
def in_bounds_or_empty(r, shape):
    if r.shape[0] == 0:
        return True
    else:
        return r.max() <= shape and r.min() >=0

def compute_batch_ap(model, dataset, config, image_ids,
                     shape=256, thresh=0.95, do_trick=True):
    APs = []

    for image_id in tqdm_notebook(image_ids):
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id,
            #    use_mini_mask=False
        )

        # print(image_meta, gt_bbox, gt_mask)
        # Run object detection
        if image_id not in preds_cache:
            results = model.detect([image], verbose=0)
            r = results[0]
            preds_cache[image_id] = r
        else:
            r = preds_cache[image_id]

        assert in_bounds_or_empty(r['rois'], shape)
        assert in_bounds_or_empty(gt_bbox, shape)
        # assert gt_bbox.shape[0] == 0 or gt_bbox.max() <= shape
        if do_trick:
            r['scores2'] = r['scores'] * ratio
            ratio = get_yhat(image_id)
        else:
            r['scores2'] = r['scores']
        rois = r['rois'][np.where(r['scores2'] > thresh)]
        APs.append(
            average_precision_image(rois, r['scores2'], gt_bbox, shape=shape)
        )
    print("mAP @ IoU=50: ", np.mean(APs))
    return dict(zip(image_ids, APs))  # print(len(results))


coco_weight_path = '/home/paperspace/Mask_RCNN/mask_rcnn_coco.h5'
def train_from_ckpt(dataset_train, dataset_val, coco_weight_path=coco_weight_path):
    assert Path(coco_weight_path).exists()
    # MAKE SURE YOU ARE ONLY EXCLUDING THE PRETRAINED WEIGHTS THAT YOU WANT
    EXCLUDE_GROUPS = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)
    model.load_weights(coco_weight_path, by_name=True)
    LEARNING_RATE = 0.005
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE/ 10,
                epochs=32,
                layers='all',
                augmentation=augmentation)


def oof_from_coco_script(weight_paths=None, N_SPLITS = 4):
    image_fps_list = np.array(list(image_fps))
    random.seed(42)
    random.shuffle(image_fps_list)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    for train_fp, val_fp in tqdm(kf.split(image_fps), total=N_SPLITS):
        image_fps_train = image_fps_list[train_fp]
        dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
        dataset_train.prepare()

        image_fps_val = image_fps_list[val_fp]
        dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
        dataset_val.prepare()

        if weight_paths is None:
            train_from_coco(dataset_train, dataset_val)
        else:
            train_from_ckpt(dataset_train, dataset_val, weight_paths[i])

#mask_rcnn_history:
# news = model.keras_model.history.history
# for k in news: history[k] = history[k] + news[k]

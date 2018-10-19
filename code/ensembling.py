import numpy as np



import pickle
def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_model_for_inference(model_path, config):
    model = modellib.MaskRCNN(mode='inference',
                              config=config,
                              model_dir=ROOT_DIR)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model

def counter(ser):
    return ser.str.count(' ').fillna(0) /5.

def to_str(boxes, score=.95):
    if np.isnan(boxes).any(): return ''
    base = "{} {} {} {} {} "
    strs = []
    for (x1, y1, w, h) in boxes:
#         w = x2 - x1
#         h = y2 - y1
        strs.append(base.format(score, x1, y1, w, h))
    return ''.join(strs)


# import the necessary packages
import numpy as np


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
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def run_nms(catted, nms_tresh=0.7):
    new_boxes = {}

    for k, v in catted.iterrows():
        box_lst = []
        for val in v.dropna().values:
            detections = chunks(val.split(), 5)
            for detection in detections:
                box_lst.append([float(x) for x in detection[1:]])
        if len(box_lst) > 0:
            new_boxes[k] = non_max_suppression_fast(np.array(box_lst), nms_tresh)
        else:
            new_boxes[k] = np.nan
    return new_boxes

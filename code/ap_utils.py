import numpy as np
import skimage
# Define auxiliary metric functions

min_box_area = 10000


def box_mask(box, shape):
    """ Create a square mask for a box from its coordinates
    :param box: [x, y, w, h] box coordinates
    :param shape: shape of the image (default set to maximum possible value, set to smaller to
    save memory)
    :returns: (np.array of bool) mask as binary 2D array
    """
    x, y, w, h = box
    mask = np.zeros((shape, shape), dtype=bool)
    mask[y:y + h, x:x + w] = True
    return mask




# # debug code for above function
# plt.imshow(box_mask([5,20,50,100], shape=256), cmap=mpl.cm.jet)

# define function that extracts confidence and coordinates of boxes from a prediction mask
def torch_mask_to_boxes(msk, threshold=0.20, connectivity=None):
    """
    :param msk: (torch.Tensor) CxWxH tensor representing the prediction mask
    :param threshold: threshold in the range 0-1 above which a pixel is considered a positive target
    :param connectivity: connectivity parameter for skimage.measure.label segmentation (can be
    None, 1, or 2)
                         http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage
                         .measure.label
    :returns: (list, list) predicted_boxes, confidences
    """
    # extract 2d array
    msk = msk[0]
    # select pixels above threshold and mark them as positives (1) in an array of equal size as
    # the input prediction mask
    pos = np.zeros(msk.shape)
    pos[msk > threshold] = 1.
    # label regions
    lbl = skimage.measure.label(pos, connectivity=connectivity)

    predicted_boxes = []
    confidences = []
    # iterate over regions and extract box coordinates
    for region in skimage.measure.regionprops(lbl):
        # retrieve x, y, height and width and add to prediction string
        y1, x1, y2, x2 = region.bbox
        h = y2 - y1
        w = x2 - x1
        c = np.nanmean(msk[y1:y2, x1:x2])
        # add control over box size (eliminate if too small)
        if w * h > min_box_area:
            predicted_boxes.append([x1, y1, w, h])
            confidences.append(c)

    return predicted_boxes, confidences


# # debug code for above function
# plt.imshow(dataset_train[3][1][0], cmap=mpl.cm.jet)
# print(dataset_train[3][1].shape)
# print(parse_boxes(dataset_train[3][1]))

def make_prediction_string(predicted_boxes, confidences):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes
    coordinates
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :returns: prediction string 'c1 x1 y1 w1 h1 c2 x2 y2 w2 h2 ...'
    """
    prediction_string = ''
    for c, box in zip(confidences, predicted_boxes):
        prediction_string += ' ' + str(c) + ' ' + ' '.join([str(b) for b in box])
    return prediction_string[1:]


# # debug code for above function
# predicted_boxes, confidences = parse_boxes(dataset_train[3][1])
# print(predicted_boxes, confidences)
# print(prediction_string(predicted_boxes, confidences))
def box_mask_coords(box, shape):
    """ Create a square mask for a box from its coordinates
    :param box: [x, y, x2, y2] box coordinates
    :param shape: shape of the image (default set to maximum possible value, set to smaller to
    save memory)
    :returns: (np.array of bool) mask as binary 2D array
    """
    x, y, x2, y2 = box
    mask = np.zeros((shape, shape), dtype=bool)
    mask[y:y2, x:x2] = True
    return mask

def IoU(pr, gt):
    """
    :param pr: (numpy_array(bool)) prediction array
    :param gt: (numpy_array(bool)) ground truth array
    :returns: IoU (pr, gt) = intersection (pr, gt) / union (pr, gt)
    """
    IoU = (pr & gt).sum() / ((pr | gt).sum() + 1.e-9)
    return IoU


# # debug code for above function
# pr = box_mask([50,60,100,150], shape=256)
# gt = box_mask([30,40,100,140], shape=256)
# plt.imshow(pr, cmap=mpl.cm.Reds, alpha=0.3)
# plt.imshow(gt, cmap=mpl.cm.Greens, alpha=0.3)
# print(IoU(pr, gt))

# define precision function
def precision(tp, fp, fn):
    """
    :param tp: (int) number of true positives
    :param fp: (int) number of false positives
    :param fn: (int) number of false negatives
    :returns: precision metric for one image at one threshold
    """
    return float(tp) / (tp + fp + fn)


# # debug code for above function
# print(precision(3,1,1))

import mrcnn.utils as utils

def compute_aps(detections, dataset, config, preds_cache={},
                shape=256, thresh=0.95, just_stats=True):
    APs = []
    n_rois = []
    image_ids = []
    n_bbox = []
    for image_id, r in tqdm_notebook(detections.items()):
        image_ids.append(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id,
            use_mini_mask=False
        )
        mask = np.where(r['scores'] > thresh)
        rois = r['rois'][np.where(r['scores'] > thresh)].astype(int)
        if (len(rois) + len(gt_bbox)) == 0:
            ap = np.nan
        elif len(rois) == 0:
            ap = 0.0
            # if we have target boxes but no predicted boxes, precision is 0
        elif len(gt_bbox) == 0:
            ap = 0.0
        else:
            masks = r['masks'][:, :, mask[0]]
            scores = r['scores'][mask]
            class_ids = np.ones(rois.shape[0])
            ap = utils.compute_ap_range(
                gt_bbox, gt_class_id, gt_mask,
                rois, class_ids, scores, masks,
                iou_thresholds=list(np.arange(0.4, .8, 0.05)),
                verbose=0
            )

        APs.append(
            # average_precision_image(rois, r['scores'], gt_bbox, shape=shape)
            ap
            # average_precision_image(rois, r['scores'], gt_bbox, shape=shape)
        )
        n_rois.append(rois.shape[0])
        n_bbox.append(gt_bbox.shape[0])
    print("mAP {:.4f} ".format(np.nanmean(APs)))
    print("mean n rois {:.2f} ".format(np.mean(n_rois)))
    return pd.DataFrame({'ap': APs, 'n_roi': n_rois, 'n_bbox': n_bbox}, index=image_ids)

def average_precision_image(predicted_boxes, confidences, target_boxes, shape=256,
                            thresholds=np.arange(0.4, 0.8, 0.05)):
    """expects [x1, y1, w1, h1]
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes
    coordinates
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :param target_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of target boxes coordinates
    :param shape: shape of the boolean masks (default set to maximum possible value,
    set to smaller to save memory)
    :returns: average_precision
    """
    # if both predicted and target boxes are empty, precision is NaN (and doesn't count towards
    # the average)

    if (len(predicted_boxes) + len(target_boxes)) == 0:
        return np.nan
    elif len(predicted_boxes) == 0:
        return 0.0
        # if we have target boxes but no predicted boxes, precision is 0
    elif len(target_boxes) == 0:
        return 0.0
    # if we have both predicted and target boxes, proceed to calculate image average precision
    else:
        # sort boxes according to their confidence (from largest to smallest)
        predicted_boxes_sorted = [b for _, b in sorted(zip(confidences, predicted_boxes),
                                                       key=lambda pair: -pair[0])]
        pred_masks = [box_mask_coords(box, shape) for box in predicted_boxes_sorted]
        targ_masks = [box_mask_coords(box, shape) for box in target_boxes]
        average_precision = 0.0
        for t in thresholds:  # iterate over thresholds
            prec = calc_precision_at_thresh(pred_masks, targ_masks, t)
            assert 0 <= prec <= 1
            average_precision += prec / float(len(thresholds))


        return average_precision


def calc_precision_at_thresh(predicted_boxes_sorted, target_boxes, t):
    n_targ_boxes = len(target_boxes)
    tp = 0  # initiate number of true positives
    fp = len(predicted_boxes_sorted)  # initiate number of false positives
    fn_mask = np.ones(len(target_boxes))
    for box_p_msk in predicted_boxes_sorted:  # iterate over predicted boxes coordinates
        for i, box_t_msk in enumerate(target_boxes):  # iterate over ground truth boxes coordinates
            iou = IoU(box_p_msk, box_t_msk)  # calculate IoU. Could hoist to save time.
            if iou > t:
                tp += 1  # if IoU is above the threshold, we got one more true positive
                fn_mask[i] = 0
                fp -= 1 # and one less false positive
                break # proceed to the next predicted box

    fn = sum(fn_mask)
    print(t, tp, fp, fn)

    prec = precision(tp, fp, fn)
    return prec


from tqdm import tqdm_notebook



def in_bounds_or_empty(r, shape):
    if r.shape[0] == 0:
        return True
    else:
        return r.max() <= shape and r.min() >= 0

def compute_aps(detections, dataset, config, shape=256, thresh=0.95):
    """Built in mask-rcnn utils assume there is a class in every image."""
    import warnings
    import mrcnn.model as modellib
    warnings.filterwarnings('ignore')   # elementwise compare failed everytime run average_precision image
    APs = []
    n_rois = []
    image_ids = []
    n_bbox = []
    for image_id in tqdm_notebook(detections.keys()):
        image_ids.append(image_id)
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id,
            use_mini_mask=False
        )
        r = detections[image_id]
        rois = r['rois'][np.where(r['scores'] > thresh)].astype(int)
        APs.append(
            average_precision_image(rois, r['scores'], gt_bbox, shape=shape)
        )
        n_rois.append(rois.shape[0])
        n_bbox.append(gt_bbox.shape[0])
    print("mAP {:.4f} ".format(np.nanmean(APs)))
    print("mean n rois {:.2f} ".format(np.mean(n_rois)))
    return pd.DataFrame({'ap': APs, 'n_roi': n_rois, 'n_bbox': n_bbox}, index=image_ids)


#res = {.5: .1417, .75: .1407, .85: .1365}
def thresh_map_check(model, dataset_val, inference_config, n=6, start=.5, end=.99):
    cache = {}
    res = {}
    for thresh in np.linspace(start, end, n):
        res[thresh] = compute_aps(model, dataset_val, inference_config,
                                       dataset_val.image_ids,
                                       preds_cache=cache,
                                       thresh=thresh, just_stats=True)
    return res


def aps_to_mean_ser(res):
    return pd.Series({k: np.mean(v) if isinstance(v, list) else v for k,v in res.items()})
import pandas as pd
from collections import defaultdict

HISTORY = defaultdict(list)

def update_history(model):
    for k,v in model.keras_model.history.history.items():
        HISTORY[k] += v

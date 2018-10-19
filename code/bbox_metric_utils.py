import numpy as np
import skimage
# Define auxiliary metric functions

min_box_area = 10000

# define function that creates a square mask for a box from its coordinates
def box_mask(box, shape=1024):
    """
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
def parse_boxes(msk, threshold=0.20, connectivity=None):
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

# define function that creates prediction strings as expected in submission
def prediction_string(predicted_boxes, confidences):
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

# define iou function
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
    return float(tp) / (tp + fp + fn + 1.e-9)


# # debug code for above function
# print(precision(3,1,1))

# define function that calculates the average precision of an image
def average_precision_image(predicted_boxes, confidences, target_boxes, shape=1024):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes
    coordinates
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :param target_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of target boxes coordinates
    :param shape: shape of the boolean masks (default set to maximum possible value,
    set to smaller to save memory)
    :returns: average_precision
    """

    # if both predicted and target boxes are empty, precision is NaN (and doesn't count towards
    # the batch average)
    if predicted_boxes == [] and target_boxes == []:
        return np.nan
    else:
        # if we have predicted boxes but no target boxes, precision is 0
        if len(predicted_boxes) > 0 and target_boxes == []:
            return 0.0
        # if we have target boxes but no predicted boxes, precision is 0
        elif len(target_boxes) > 0 and predicted_boxes == []:
            return 0.0
        # if we have both predicted and target boxes, proceed to calculate image average precision
        else:
            # define list of thresholds for IoU [0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75]
            thresholds = np.arange(0.4, 0.8, 0.05)
            # sort boxes according to their confidence (from largest to smallest)
            predicted_boxes_sorted = list(
                reversed([b for _, b in sorted(zip(confidences, predicted_boxes),
                                               key=lambda pair: pair[0])]))
            average_precision = 0.0
            for t in thresholds:  # iterate over thresholds
                # with a first loop we measure true and false positives
                tp = 0  # initiate number of true positives
                fp = len(predicted_boxes)  # initiate number of false positives
                for box_p in predicted_boxes_sorted:  # iterate over predicted boxes coordinates
                    box_p_msk = box_mask(box_p, shape)  # generate boolean mask
                    for box_t in target_boxes:  # iterate over ground truth boxes coordinates
                        box_t_msk = box_mask(box_t, shape)  # generate boolean mask
                        iou = IoU(box_p_msk, box_t_msk)  # calculate IoU
                        if iou > t:
                            tp += 1  # if IoU is above the threshold, we got one more true positive
                            fp -= 1  # and one less false positive
                            break  # proceed to the next predicted box
                # with a second loop we measure false negatives
                fn = len(target_boxes)  # initiate number of false negatives
                for box_t in target_boxes:  # iterate over ground truth boxes coordinates
                    box_t_msk = box_mask(box_t, shape)  # generate boolean mask
                    for box_p in predicted_boxes_sorted:  # iterate over predicted boxes coordinates
                        box_p_msk = box_mask(box_p, shape)  # generate boolean mask
                        iou = IoU(box_p_msk, box_t_msk)  # calculate IoU
                        if iou > t:
                            fn -= 1
                            break  # proceed to the next ground truth box
                # TBD: this algo must be checked against the official Kaggle evaluation method
                # which is still not clear...
                average_precision += precision(tp, fp, fn) / float(len(thresholds))
            return average_precision


# # debug code for above function
# confidences = [0.3, 0.9]
# predicted_boxes = [[20,20,60,70], [110,110,50,70]]
# target_boxes = [[25,25,60,70], [100,100,50,70]]#, [200, 200, 30, 50]]
# for box_p in predicted_boxes:
#     plt.imshow(box_mask(box_p, shape=256), cmap=mpl.cm.Reds, alpha=0.3)
# for box_t in target_boxes:
#     plt.imshow(box_mask(box_t, shape=256), cmap=mpl.cm.Greens, alpha=0.3)
# print(average_precision_image(predicted_boxes, confidences, target_boxes))

# define function that calculates the average precision of a batch of images
def average_precision_batch(output_batch, pIds, pId_boxes_dict, rescale_factor, shape=1024,
                            return_array=False):
    """
    :param output_batch: cnn model output batch
    :param pIds: (list) list of patient IDs contained in the output batch
    :param rescale_factor: CNN image rescale factor
    :param shape: shape of the boolean masks (default set to maximum possible value,
    set to smaller to save memory)
    :returns: average_precision
    """

    batch_precisions = []
    for msk, pId in zip(output_batch,
                        pIds):  # iterate over batch prediction masks and relative patient IDs
        # retrieve target boxes from dictionary (quicker than from mask itself)
        target_boxes = pId_boxes_dict[pId] if pId in pId_boxes_dict else []
        # rescale coordinates of target boxes
        if len(target_boxes) > 0:
            target_boxes = [[int(round(c / float(rescale_factor))) for c in box_t] for box_t in
                            target_boxes]
        # extract prediction boxes and confidences
        predicted_boxes, confidences = parse_boxes(msk)
        batch_precisions.append(
            average_precision_image(predicted_boxes, confidences, target_boxes, shape=shape))
    if return_array:
        return np.asarray(batch_precisions)
    else:
        return np.nanmean(np.asarray(batch_precisions))

def compute_batch_ap(model, dataset, config, image_ids, preds_cache={},
                     shape=256, thresh=0.95, just_stats=True):
    import warnings
    warnings.filterwarnings('ignore')
    APs = []
    dets = []
    for image_id in tqdm_notebook(image_ids):
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id,
            #    use_mini_mask=False
        )
        if image_id not in preds_cache:
            r = model.detect([image], verbose=0)[0]
            preds_cache[image_id] = r
        else:
            r = preds_cache[image_id]
        rois = r['rois'][np.where(r['scores'] > thresh)]
        APs.append(
            average_precision_image(rois, r['scores'], gt_bbox, shape=shape)
        )
        dets.append(r)
    print("mAP @ IoU=50: {:.4f} ".format(np.mean(APs)))
    if just_stats:
        return APs
    else:
        return zip(image_ids, APs, dets)


def thresh_map_check(model, dataset_val, inference_config, n=6, start=.5):
    cache = {}
    for thresh in np.linspace(.5, 1., n=n):
        res[thresh] = compute_batch_ap(model, dataset_val, inference_config,
                                       dataset_val.image_ids,
                                       preds_cache=cache,
                                       thresh=thresh, just_stats=True)
    return res


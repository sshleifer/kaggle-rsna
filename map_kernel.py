import numpy as np


# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores,
            thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[
                                           1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


IOU_FULL = list(np.arange(0.4, .8, 0.05))
IOU_FAST = [.6]
def convert_to_w_h(bbox):
    new_arr = bbox.copy()
    new_arr[:, 2] = new_arr[:, 2] - new_arr[:, 0]
    new_arr[:, 3] = new_arr[:, 3] - new_arr[:, 1]
    return new_arr

def shuffle_x_y(g):
    return np.hstack([g[:, 1], g[:, 0], g[:, 3], g[:, 2]]).reshape(-1, 4)

def show_image2(image_id, boxes, ax=None, display_thresh=0.9):
    resize_factor = 4
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image(dataset_val, inference_config, image_id)
    resize_factor = original_image.shape[0] / 256.
    for i, bbox in enumerate(boxes):
        print(bbox)
        score = r['scores'][i]
        if score < display_thresh:
            break
        if len(bbox) == 6:
            bbox = bbox[2:]
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2] * resize_factor)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (77, 255, 9), 3, 1)
    ax = show_img(original_image)
    return ax

def compute_ap2(detections, dataset=dataset_val, config=inference_config,
                iou_thresholds=IOU_FAST,
                just_stats=False, debug=True,
                thresh=0., shape=256):
    APs = []
    n_rois = []
    image_ids = []
    n_bbox = []
    for patient_id, r in list(detections.items()):
        image_ids.append(patient_id)
        gt_bbox = load_bbox(dataset, config, patient_id)
        gt_bbox2 = convert_to_w_h(gt_bbox) * (1024 / shape)

        arr = np.array([x for x in r if x[-1] > thresh])
        if (len(arr) + len(gt_bbox)) == 0:
            ap = np.nan
        elif len(arr) == 0:
            ap = 0.0
            # if we have target boxes but no predicted boxes, precision is 0
        elif len(gt_bbox) == 0:
            ap = 0.0
        else:
            if debug:
                ax = show_image2(PATIENT_TO_IID[patient_id], gt_bbox)
                plt.show()
                break
            scores = arr[:, -1]
            gt_bbox3 = shuffle_x_y(gt_bbox2)
            ap = map_iou(gt_bbox3, arr[:, :4], scores,
                         thresholds=iou_thresholds)
        APs.append(ap)
        n_rois.append(arr.shape[0])
        n_bbox.append(gt_bbox.shape[0])
    if just_stats:
        return dict(ap=np.nanmean(APs), n_rois=np.mean(n_rois))
    print("mAP {:.4f} ".format(np.nanmean(APs)))
    print("mean n rois {:.2f} ".format(np.mean(n_rois)))
    stat_df = pd.DataFrame({'ap': APs, 'n_roi': n_rois, 'n_bbox': n_bbox}, index=image_ids)
    return stat_df




    scores_arr = np.array(lmap(get_top_proba, sub_df)

PATH_600_300 = '/home/paperspace/keras-retinanet/sub_retnet_600_300_resnet50_07.csv'
PATH_320_320 = '/home/paperspace/keras-retinanet/sub_retnet_320_320_resnet50_09.csv'
PATH_320_320 = '/home/paperspace/keras-retinanet/sub_retnet_300_400_resnet50_04.csv'
PATH_320_320 = '/home/paperspace/keras-retinanet/sub_retnet_256_256_resnet50_09.csv'
PATH_200_200 = '/home/paperspace/keras-retinanet/sub_retnet_200_200_resnet50_07.csv'

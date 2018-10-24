def shape_ser(outs50):
    return pd.Series({k: len(v) for k,v in outs50.items()})
dataset_val = pickle_load('dataset_val.pkl')

mrcnn_val_dets = pickle_load('vr101.pkl')

rnet_dets_val = pickle_load('/home/paperspace/keras-retinanet/dets_val_filtered.pkl')

rnet_dets_te = pickle_load('/home/paperspace/keras-retinanet/dets_te_filtered.pkl')
rnet_te_image_paths = pickle_load('/home/paperspace/keras-retinanet/te_gen_image_names.pkl')
rnet_val_img_paths = pickle_load('/home/paperspace/keras-retinanet/val_gen_image_names.pkl')
r50_val = mrcnn_val_dets['res50']
r50_te_dets = pickle_load('resnet_50_te_preds.pkl')
C = np.concatenate
def path_to_patient_id(path):
    return os.path.splitext(os.path.basename(path))[0]

def reshape_retinanet(d):
    d = convert_to_w_h(d[0])
    class_ids = np.ones((d.shape[0], 1))
    scores = d[:, -1:]
    return C([d[:, :4], class_ids, scores], axis=1)
def convert_to_w_h(bbox):
    new_arr = bbox.copy()
    new_arr[:, 2] = new_arr[:, 2] - new_arr[:, 0]
    new_arr[:, 3] = new_arr[:, 3] - new_arr[:, 1]
    return new_arr
def GeneralEnsemble(dets, iou_thresh=0.5, weights=None):
    """
    General Ensemble - find overlapping boxes of the same class and average their positions
    while adding their confidences. Can weigh different detectors with different weights.
    No real learning here, although the weights and iou_thresh can be optimized.
    Input:
     - dets : List of detections. Each detection is all the output from one detector, and
              should be a list of boxes, where each box should be on the format
              [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y
              are the center coordinates, box_w and box_h are width and height resp.
              The values should be floats, except the class which should be an integer.
     - iou_thresh: Threshold in terms of IOU where two boxes are considered the same,
                   if they also belong to the same class.

     - weights: A list of weights, describing how much more some detectors should
                be trusted compared to others. The list should be as long as the
                number of detections. If this is set to None, then all detectors
                will be considered equally reliable. The sum of weights does not
                necessarily have to be 1.
    Output:
        A list of boxes, on the same format as the input. Confidences are in range 0-1.
    """
    assert (type(iou_thresh) == float)

    ndets = len(dets)

    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    else:
        assert (len(weights) == ndets), (len(weights), ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()

    for idet in range(0, ndets):
        det = dets[idet]
        # print(det)
        for box in det:
            if box in used:
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w * b[0]
                    yc += w * b[1]
                    bw += w * b[2]
                    bh += w * b[3]
                    conf += w * b[5]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out


def getCoords(box):
    x1 = float(box[0]) - float(box[2]) / 2
    x2 = float(box[0]) + float(box[2]) / 2
    y1 = float(box[1]) - float(box[3]) / 2
    y2 = float(box[1]) + float(box[3]) / 2
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou

def do_all_val(ret_dct_val, r50_val, iou_thresh=.99):
    outs = {}
    for k, v in list(ret_dct_val.items()):
        rcnn_det = r50_val[k]
        if isinstance(v, np.ndarray):
            v = v.tolist()
        dets = [v, rcnn_det.tolist()]
        # print(dets)
        outs[k] = GeneralEnsemble(dets, iou_thresh=iou_thresh)
    return outs


def nms_on_retnet(ret_dct_val, iou_thresh=.1):
    outs = {}
    for k, v in list(ret_dct_val.items()):
        dets = [[box] for box in v.tolist()]
        if len(dets) == 0:
            outs[k] = []
            continue
        outs[k] = GeneralEnsemble(dets, iou_thresh=iou_thresh)
    return outs


def rcnn_reshaper(r, thresh, scaler=1024 / 256):
    mask = np.where(r['scores'] > thresh)
    rois = r['rois'][mask].astype(int)
    rois = convert_to_w_h(rois * scaler)
    new_shape = (rois.shape[0], 1)

    return C([
        rois, r['class_ids'][mask].reshape(new_shape),
        normalize_scores(r['scores'][mask]).reshape(new_shape)
    ], axis=1)

def normalize_scores(scores):
    return ((scores - 0.8) * 10).clip(0, )

def dhead(d, n=5): return funcy.project(d, funcy.take(n, d.keys()))
mrcnn_val_paths = [x['path'] for x in dataset_val.image_info]
ret_val_img_patients = lmap(path_to_patient_id, rnet_val_img_paths)
mrcnn_val_img_patients = lmap(path_to_patient_id, mrcnn_val_paths)


r50_te = {path_to_patient_id(k): rcnn_reshaper(x, .95)
          for k, x in r50_te_dets.items()}
shape_ser(r50_te).value_counts()
reshaped_r50_val = [rcnn_reshaper(x, .95) for x in r50_val]
r50_val = dict(zip(mrcnn_val_img_patients, reshaped_r50_val))

rnet_dets_val = pickle_load('/home/paperspace/keras-retinanet/dets_val_filtered.pkl')
rnet_dets_te = pickle_load('/home/paperspace/keras-retinanet/dets_te_filtered.pkl')
rnet_te_image_paths = pickle_load('/home/paperspace/keras-retinanet/te_gen_image_names.pkl')
rnet_val_img_paths = pickle_load('/home/paperspace/keras-retinanet/val_gen_image_names.pkl')
ret_reshaped_val = lmap(reshape_retinanet, rnet_dets_val)
ret_reshaped_te = lmap(reshape_retinanet, rnet_dets_te)

path_to_patient_id(rnet_val_img_paths[0])

ret_val_img_patients = lmap(path_to_patient_id, rnet_val_img_paths)
mrcnn_val_img_patients = lmap(path_to_patient_id, mrcnn_val_paths)

len(set(ret_val_img_patients).intersection(mrcnn_val_img_patients))

ret_dct_val = dict(zip(ret_val_img_patients, ret_reshaped_val))

ret_dct_te = {path_to_patient_id(p): x for p, x in zip(rnet_te_image_paths, ret_reshaped_te)}


# len(ret_dct_te)



outs_retnet = nms_on_retnet(ret_dct_val, iou_thresh=.01)

outs50 = do_all_val(outs_retnet, r50_val, iou_thresh=.01)

outs_retnet_te = nms_on_retnet(ret_dct_te, iou_thresh=.01)

shape_ser(ret_dct_te).mean()

shape_ser(outs_retnet_te).mean()



def preprocess_retinanet(path, skip_box_thr, iou_thr,ret_val_img_patients=ret_val_img_patients):
    val_rnet_256 = lmap(lambda x: reshape_retinanet(x, skip_box_thr),
                        pickle_load(path))
    val_rnet_256_dct = dict(zip(ret_val_img_patients, val_rnet_256))
    output = nms_on_retnet(val_rnet_300_dct, .01)
    compute_ap2(output)
    return output


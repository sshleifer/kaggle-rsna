


dets_to_use = [



]
feat_df = pd.concat([
    make_features(vr101['res50'], 'res50'),
    make_features(vr101['res101'], 'res101'),
    make_features(mend_dets, 'smaller_anchors'),
    make_features(high_nms_dets, 'high_nms'),
    make_features(nih_val_dets, 'nih_tr'),
], axis=1)


import numpy as np
import pandas as pd

def featureize(detections, thresh=0.95):
    n_rois = []
    image_ids = []
    scores_lst = []
    for image_id, r in detections.items():
        image_ids.append(image_id)
        rois = r['rois'][np.where(r['scores'] > thresh)].astype(int)
        n_rois.append(rois.shape[0])
        scores_lst.append(r['scores'])
    score_df = pd.DataFrame(scores_lst, index=image_ids).add_prefix('score_')
    score_df['n_rois'] = n_rois
    return score_df


MODEL_PAIRS = [
    ('nih_negs', 'val_dets_nih_negs_mrcnn50.pkl', 'te_dets_nih_negs_mrcnn50.pkl'),
    ('high_nms', 'val_dets_high_nms_mrcnn50.pkl', 'te_dets_high_nms_mrcnn50.pkl'),
    ('small_anchors', 'mendonca_val_dets.pkl', 'mendonca_test_dets.pkl'),
    ('res50', 'validation_resnet50.pkl', 'resnet_50_te_preds.pkl'),
    ('res101', 'validation_resnet_101.pkl', 'resnet_101_te_preds.pkl'),
    ('res101b', 'val_dets_take_2_mrcnn101.pkl', 'te_dets_take_2_mrcnn101.pkl'),
]

def make_feature_df(model_pairs):
    val_dfs = []
    te_dfs = []
    for model_name, name, val_pth, te_pth in model_pairs:
        val_dfs.append(featureize(pickle_load(val_pth)).add_prefix(model_name + '_'))
        te_dfs.append(featureize(pickle_load(val_pth)).add_prefix(model_name + '_'))
    return pd.concat(val_dfs, te_dfs)

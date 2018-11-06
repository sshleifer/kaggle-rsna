from tqdm import *
import numpy as np
import funcy
from map_kernel import map_iou
from analyze_subs import lmap


st1_test_df = None


def score_pat(sub, patient):
    sub_row = sub.loc[patient]
    truth_rows = st1_test_df[st1_test_df.patientId == patient].dropna()
    truth_arr = truth_rows[['x', 'y', 'width', 'height']].values
    if isinstance(sub_row, np.float):
        if truth_rows.empty:
            return np.nan
        else:
            return 0
    elif truth_rows.empty:
        return 0
    else:
        bboxes = list(funcy.chunks(5, lmap(float, sub_row.strip().split(' '))))
        scores = np.array(bboxes)[:, 0]
        arr = np.array(bboxes)[:, 1:]
        return map_iou(truth_arr, arr, scores)


def score_sub(sub):
    aps = [score_pat(sub, pat)
           for pat in tqdm_notebook(st1_test_df.patientId.unique())]
    print('{:.3f}'.format(np.nanmean(aps)))
    return np.nanmean(aps)

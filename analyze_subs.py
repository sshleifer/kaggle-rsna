import numpy as np


def thrensemble(scores_arr, dets_arr, threshes):
    final = []
    for i in range(len(scores_arr[0])):
        for j, (score, thresh) in enumerate(zip(scores_arr[i], threshes)):
            if score > thresh:
                final.append(dets_arr[i][j])
                break
        else:
            final.append(np.nan)
    return final


def get_nth_largest_proba(ser, n=300):
    probas = get_top_proba(ser)
    return probas.nlargest(n).values[-1]


def thresh_to_nth_largest_proba(ser, n):
    probas = get_top_proba(ser)
    cutoff = get_nth_largest_proba(ser, n)
    mask = probas < cutoff
    ret_ser = ser.copy()
    ret_ser.loc[mask] = np.nan
    return ret_ser


def read_sub(path):
    return pd.read_csv(path).set_index(PATIENT_ID)[P]


def save_sub(ser, path):
    df = ser.rename_axis(PATIENT_ID).to_frame(P)
    df.to_csv(path, index=True)
    return path


def get_top_proba(yday):
    return yday.fillna('.00 ').str.strip().str.partition(' ')[0].astype(float)


def lmap(fn, coll): return list(map(fn, coll))


PATIENT_ID = 'patientId'
P = 'PredictionString'


def run_thrensemble(sub_df, threshes=[.15, .15]):
    scores_arr = sub_df.apply(get_top_proba).values
    dets_arr = sub_df.values
    return thrensemble(scores_arr, dets_arr, threshes)


import funcy


def get_1_or_2(x, cut1, cut2):
    """Unused"""
    if len(x) == 0:
        return np.nan
    new_dets = []
    x = lmap(float, x)
    chunks = list(funcy.chunks(5, x))
    for i, c in enumerate(chunks):
        if c[0] > cut1:
            new_dets.append(c)
        elif (i == 1) and c[0] > cut2:
            assert len(new_dets) == 1
            new_dets.append(c)
        else:
            break
    return ' '.join(list(funcy.flatten(new_dets)))


def make_1_or_2_ser(ser, cut1, cut2):
    splat = ser.fillna(' ').str.strip().str.split(' ')
    dets = [get_1_or_2(x, cut1, cut2) for x in splat.values]
    return pd.Series(dets, index=ser.index)

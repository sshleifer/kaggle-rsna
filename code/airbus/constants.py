from pathlib import Path

PATH = Path('/home/paperspace/')


def abs_path(x): return PATH / x


DATA_DIR = PATH / 'airbus_data/'
TRAIN = DATA_DIR / 'train_v2/'
TEST = DATA_DIR / 'test_v2/'
SEGMENTATION = DATA_DIR / 'train_ship_segmentations_v2.csv'
PRETRAINED_SEGMENTATION_PATH = PATH / 'lafoss_ckpt'
DETECTION_TEST_PRED = DATA_DIR / 'ship_detection.csv'
PSHIP = 'p_ship'


import pickle
def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
save_pickle = pickle_save
load_pickle = pickle_load
import funcy
def denumerate(lst): return dict(enumerate(lst))
def lmap(fn, coll): return list(map(fn, coll))
def dhead(d, n=5): return funcy.project(d, funcy.take(n, d.keys()))

tr_n_cut = pickle_load(DATA_DIR / 'train_n_cut95.pkl')
val_n_cut = pickle_load(DATA_DIR/ 'val_n_cut95.pkl')
tr_n_cut98 = pickle_load(DATA_DIR / 'train_n_cut98.pkl')
val_n_cut98 = pickle_load(DATA_DIR/ 'val_n_cut98.pkl')


IMAGE_ID ='ImageId'
ENC_PIX ='EncodedPixels'
import pandas as pd
def read_sub(path):
    return pd.read_csv(path).set_index(IMAGE_ID)[ENC_PIX]
def read_probas(path):
    return pd.read_csv(path).set_index('id').p_ship

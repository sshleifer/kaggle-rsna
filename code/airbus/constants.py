from pathlib import Path

PATH = Path('/home/paperspace/')


def abs_path(x): return PATH / x


DATA_DIR = PATH / 'airbus_data/'
TRAIN = DATA_DIR / 'train_v2/'
TEST = DATA_DIR / 'test_v2/'
SEGMENTATION = DATA_DIR / 'train_ship_segmentations_v2.csv'
PRETRAINED_SEGMENTATION_PATH = PATH / 'lafoss_ckpt'
DETECTION_TEST_PRED = DATA_DIR / 'ship_detection.csv'

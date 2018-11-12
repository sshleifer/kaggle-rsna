import os
import pandas as pd
import cv2

from .constants import *
from fastai.dataset import FilesDataset
from fastai.dataset import *


def get_mask(img_id, df, shape=(768, 768)):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if (type(masks) == float): return img.reshape(shape)
    if (type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1
    return img.reshape(shape).T


class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform, seg_path=SEGMENTATION):
        self.segmentation_df = pd.read_csv(seg_path).set_index('ImageId')
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768:
            return img
        else:
            return cv2.resize(img, (self.sz, self.sz))

    def get_y(self, i):
        mask = np.zeros((768, 768), dtype=np.uint8) if (self.path == TEST) \
            else get_mask(self.fnames[i], self.segmentation_df)
        img = Image.fromarray(mask).resize((self.sz, self.sz)).convert('RGB')
        return np.array(img).astype(np.float32)

    def get_c(self):
        return 0


test_names = []  # global from NB
TRAIN_TFMS = [
    RandomRotate(20, tfm_y=TfmType.CLASS),
    RandomDihedral(tfm_y=TfmType.CLASS),
    RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)
]


def get_data(sz, bs, test_names=test_names, n_val=None, n_train=None, aug_tfms=TRAIN_TFMS,
             seg_path=SEGMENTATION):
    if n_val is None:
        val_data = val_n_cut
    else:
        val_data = val_n_cut[:n_val]
    if n_train is None:
        train_data = tr_n_cut
    else:
        train_data = tr_n_cut[:n_train]

    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS,
                           aug_tfms=aug_tfms)
    # cut incomplete batch
    tr_names = train_data if (len(train_data) % bs == 0) else train_data[:-(len(train_data) % bs)]

    ds = ImageData.get_ds(
        pdFilesDataset, (tr_names, TRAIN), (val_data, TRAIN), tfms, test=(test_names, TEST),
        seg_path=seg_path
    )
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    return md


def big_boy_get_data(sz, bs, train_data, val_data, test_names=test_names, aug_tfms=TRAIN_TFMS,
                     seg_path=SEGMENTATION):
    arch = resnet34
    nw = 8
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS,
                           aug_tfms=aug_tfms)
    # cut incomplete batch
    tr_names = train_data if (len(train_data) % bs == 0) else train_data[:-(len(train_data) % bs)]

    ds = ImageData.get_ds(
        pdFilesDataset, (tr_names, TRAIN), (val_data, TRAIN), tfms, test=(test_names, TEST),
        seg_path=seg_path
    )
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    return md

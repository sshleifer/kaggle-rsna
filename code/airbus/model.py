import numpy as np
import pandas as pd
from fastai.dataset import *
from scipy import ndimage

from .constants import SEGMENTATION

cut,lr_cut = 8,6


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:, 0]

    def close(self):
        for sf in self.sfs: sf.remove()


class UnetModel():
    def __init__(self, model, name='Unet'):
        self.model, self.name = model, name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]


def IoU(pred, targs):
    pred = (pred > 0.5).astype(float)
    intersection = (pred * targs).sum()
    return intersection / ((pred + targs).sum() - intersection + 1.0)


def get_score(pred, true):
    """Higher is better."""
    n_th = 10
    b = 4
    thresholds = [0.5 + 0.05 * i for i in range(n_th)]
    n_masks = len(true)
    n_pred = len(pred)
    if (n_masks == 0) and (n_pred ==0):
        return 1.
    elif (n_masks == 0) and (n_pred > 0):
        return 0.
    ious = []
    score = 0
    for mask in true:
        buf = []
        for p in pred: buf.append(IoU(p, mask))
        ious.append(buf)
    for t in thresholds:
        tp, fp, fn = 0, 0, 0
        for i in range(n_masks):
            match = False
            for j in range(n_pred):
                if ious[i][j] > t: match = True
            if not match: fn += 1

        for j in range(n_pred):
            match = False
            for i in range(n_masks):
                if ious[i][j] > t: match = True
            if match:
                tp += 1
            else:
                fp += 1
        score += ((b + 1) * tp) / ((b + 1) * tp + b * fn + fp)
    return score / n_th

from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import funcy

def split_mask(mask, threshold = 0.5, threshold_obj = 30,
               transform_func=None, disk_size=3):
    selem = disk(disk_size)
    if transform_func is not None:
        transformed_mask = transform_func(mask, selem)
    else:
        transformed_mask = mask
    labled, n_objs = ndimage.label(transformed_mask > threshold)
    result = []
    # ignore predictions composed of "threshold_obj" pixels or less
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if (obj.sum() > threshold_obj): result.append(obj)
    return result


def get_mask_ind(img_id, df, shape=(768, 768)):  # return mask for each ship
    masks = df.loc[img_id]['EncodedPixels']
    if (type(masks) == float): return []
    if (type(masks) == str): masks = [masks]
    result = []
    for mask in masks:
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        s = mask.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1
        result.append(img.reshape(shape).T)
    return result

def msk_sum(msk):
    comb = sum(msk)
    if isinstance(comb, int):
        return comb
    else:
        return comb.sum()

from .constants import  SEG_V3
class Score_eval():
    def __init__(self, seg_path=SEG_V3):
        self.segmentation_df = pd.read_csv(seg_path).set_index('ImageId')
        self.score, self.count = 0.0, 0
        self.masks = {}
        self.scores = {}

    def put(self, pred, name):
        true = get_mask_ind(name, self.segmentation_df)
        self.masks[name] = pred
        cur_score = get_score(pred, true)
        self.scores[name] = cur_score
        self.score += cur_score
        self.count += 1

    def evaluate(self):
        return self.score / self.count

    @property
    def score_df(self):
        n_dets = {k: len(v) for k, v in self.masks.items()}
        pix_sums = pd.Series({k: msk_sum(v) for k, v in self.masks.items()})
        df =  pd.DataFrame(dict(n_dets=n_dets, score=self.scores, pixel_sum=pix_sums))
        true_counts = self.segmentation_df.groupby(level=0).count()
        df['true_n_dets'] = true_counts
        return df





### Augmentation
def aug_unit(x, fwd=True, mask=False):
    return x


def aug_flipV(x, fwd=True, mask=False):
    return x.flip(2) if mask else x.flip(3)


def aug_flipH(x, fwd=True, mask=False):
    return x.flip(1) if mask else x.flip(2)


def aug_T(x, fwd=True, mask=False):
    return torch.transpose(x, 1, 2) if mask else torch.transpose(x, 2, 3)


def aug_rot_2(x, fwd=True, mask=False):  # rotate pi/2
    return aug_flipV(aug_flipH(x, fwd, mask), fwd, mask)


def aug_rot_4cr(x, fwd=True, mask=False):  # rotate pi/4 counterclockwise
    return aug_flipV(aug_T(x, fwd, mask), fwd, mask) if fwd else \
        aug_T(aug_flipV(x, fwd, mask), fwd, mask)


def aug_rot_4cw(x, fwd=True, mask=False):  # rotate pi/4 clockwise
    return aug_flipH(aug_T(x, fwd, mask), fwd, mask) if fwd else \
        aug_T(aug_flipH(x, fwd, mask), fwd, mask)


def aug_rot_2T(x, fwd=True, mask=False):  # transpose and rotate pi/2
    return aug_rot_2(aug_T(x, fwd, mask), fwd, mask)


trms_side_on = [aug_unit, aug_flipH]
trms_top_down = [aug_unit, aug_flipV]
trms_dihedral = [aug_unit, aug_flipH, aug_flipV, aug_T, aug_rot_2, aug_rot_2T,
                 aug_rot_4cw, aug_rot_4cr]


def enc_img(img):
    return torch.transpose(torch.tensor(img), 0, 2).unsqueeze(0)


def dec_img(img):
    return to_np(torch.transpose(img.squeeze(0), 0, 2))


def display_augs(x, augs=aug_unit):
    columns = 4
    n = len(augs)
    rows = n // 4 + 1
    fig = plt.figure(figsize=(columns * 4, rows * 4))
    img = enc_img(x)
    for i in range(rows):
        for j in range(columns):
            idx = j + i * columns
            if idx >= n: break
            fig.add_subplot(rows, columns, idx + 1)
            plt.axis('off')
            plt.imshow(dec_img(augs[idx](img)))
    plt.show()


### Inference utils

def model_pred(learner, dl, F_save):  # if use train dl, disable shuffling
    learner.model.eval();
    name_list = dl.dataset.fnames
    num_batchs = len(dl)
    t = tqdm(iter(dl), leave=False, total=num_batchs)
    count = 0
    for x, y in t:
        py = to_np(torch.sigmoid(learn.model(V(x))))
        batch_size = len(py)
        for i in range(batch_size):
            F_save(py[i], to_np(y[i]), name_list[count])
            count += 1


def pred_aug(learn, x, aug=[aug_unit]):
    pred = []
    for aug_cur in aug:
        py = to_np(aug_cur(torch.sigmoid(learn.model(V(aug_cur(x)))),
                           fwd=False, mask=True))
        pred.append(py)
    pred = np.stack(pred, axis=0).mean(axis=0)
    return pred


# if use train dl, disable shuffling
def model_pred_aug(learner, dl, F_save, aug=[aug_unit]):
    learner.model.eval();
    name_list = dl.dataset.fnames
    num_batchs = len(dl)
    t = tqdm(iter(dl), leave=False, total=num_batchs)
    count = 0
    for x, y in t:
        pred = pred_aug(learner, x, aug)
        batch_size = len(pred)
        for i in range(batch_size):
            F_save(pred[i], to_np(y[i]), name_list[count])
            count += 1


def decode_mask(mask, shape=(768, 768)):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

from .constants import pickle_load


def run_enc_test(paths, **kwargs):
    ship_list_dict = []
    for name in test_names_nothing:
        ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
    for path in paths:
        mask = pickle_load(path)
        name = r[0][:-4]
        mas
        enc_test(*r, **kwargs)
    pred_df = pd.DataFrame(ship_list_dict)
    return pred_df

def run_enc_test(flat_upr, **kwargs):
    ship_list_dict = []
    for name in test_names_nothing:
        ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
    for r in flat_upr:
        enc_test(*r, **kwargs)
    pred_df = pd.DataFrame(ship_list_dict)
    return pred_df


def analyze_sub(path):
    sub = read_sub(path)
    return pd.Series({'n_masks': sub.count(), 'n_image_id_with_mask': sub.dropna()['ImageId'].nunique()})

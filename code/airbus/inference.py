from .model import *
from .data import get_data

def run_eval(learn):
    """9 mins roughly"""
    score = Score_eval()
    sz = 768 # image size
    bs = 12  # batch size
    md = get_data(sz,bs)
    process_pred = lambda yp, y, name : score.put(split_mask(yp),name)
    model_pred_aug(learn, md.val_dl, process_pred, trms_dihedral)
    return score.evaluate()


def make_sub(learn, test_names_nothing):
    """12 mins roughly"""
    ship_list_dict = []
    for name in test_names_nothing:
        ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
    model_pred_aug(learn, md.test_dl, enc_test, trms_dihedral)
    pred_df = pd.DataFrame(ship_list_dict)
    return pred_df

def load_full_path(learn, path):
    learn.models_path = os.path.dirname(path)
    strang = os.path.splitext(os.path.basename(path))[0]
    learn.load(strang)


my_wt = [
    'Unet34_256_1.h5',
    'Unet34_384_12.h5',
    'Unet34_768_6.h5',
    'Unet34_768_1.h5',
    'Unet34_256_0.h5'
]


def enc_test(yp, y, name):
    masks = split_mask(yp)
    if(len(masks) == 0):
        ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
    for mask in masks:
        ship_list_dict.append({'ImageId':name,'EncodedPixels':decode_mask(mask)})
"""
scp -r airbus_data paperspace@184.105.174.55:./
scp -r *.ipynb paperspace@184.105.174.55:./




"""


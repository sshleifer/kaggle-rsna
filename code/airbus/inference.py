from .model import *
from fastai.conv_learner import LossRecorder
#from .data import get_data_tr

def run_eval(learn, bs = 12, n_val=None, aug_tfms=TE_TFMS):
    """9 mins roughly"""
    score = Score_eval()
    md = get_data(768, bs, n_val=n_val, aug_tfms=aug_tfms)
    process_pred = lambda yp, y, name : score.put(split_mask(yp),name)
    model_pred_aug(learn, md.val_dl, process_pred, trms_dihedral)
    return round(score.evaluate(), 5)
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
    """Intentional side effects on ship_list_dict"""
    masks = split_mask(yp)
    if(len(masks) == 0):
        ship_list_dict.append({'ImageId': name, 'EncodedPixels': np.nan})
    for mask in masks:
        ship_list_dict.append({'ImageId': name, 'EncodedPixels': decode_mask(mask)})
"""
scp -r airbus_data paperspace@184.105.174.55:./
scp -r *.ipynb paperspace@184.105.174.55:./
"""


class SaveBestModel(LossRecorder):
    def __init__(self, model, lr, name, best_loss=None):
        super().__init__(model.get_layer_opt(lr, None))
        self.name = name
        self.model = model
        self.best_loss = best_loss

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        loss, lam, dice, iou_ = metrics
        if self.best_loss == None or loss < self.best_loss:  # best model
            self.best_loss = loss
            save_path = f'{self.name}_{loss:.3f}'
            print('Saving to {}'.format(save_path))
            self.model.save(save_path)

class SaveBestDice(LossRecorder):
    def __init__(self, model, lr, name='best_model',best_loss=None):
        super().__init__(model.get_layer_opt(lr, None))
        self.name = name
        self.model = model
        self.best_loss = best_loss   # 0.850557 dice...

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        _, lam, dice, iou_ = metrics
        if self.best_loss == None or dice > self.best_loss:  # best model
            self.best_loss = dice
            save_path = f'{self.name}_{dice:.3f}'
            print('Saving to {}'.format(save_path))
            self.model.save(save_path)


def split_test_detections(ship_detection, det_thresh=0.5):
    test_names = ship_detection.loc[ship_detection['p_ship'] > det_thresh, ['id']][
        'id'].values.tolist()
    test_names_nothing = ship_detection.loc[ship_detection['p_ship'] <= det_thresh, ['id']][
        'id'].values.tolist()
    print(f' using {len(test_names)}, ignoring {len(test_names_nothing)}')
    return test_names, test_names_nothing

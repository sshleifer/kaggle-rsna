import torch
from fastai.conv_learner import *
from fastai.dataset import *

def aug_unit(x,fwd=True,mask=False):
    return x

def aug_flipV(x,fwd=True,mask=False):
    return x.flip(2) if mask else x.flip(3)

def aug_flipH(x,fwd=True,mask=False):
    return x.flip(1) if mask else x.flip(2)

def aug_T(x,fwd=True,mask=False):
    return torch.transpose(x,1,2) if mask else torch.transpose(x,2,3)

def aug_rot_2(x,fwd=True,mask=False): #rotate pi/2
    return aug_flipV(aug_flipH(x,fwd,mask),fwd,mask)

def aug_rot_4cr(x,fwd=True,mask=False): #rotate pi/4 counterclockwise
    return aug_flipV(aug_T(x,fwd,mask),fwd,mask) if fwd else \
        aug_T(aug_flipV(x,fwd,mask),fwd,mask)

def aug_rot_4cw(x,fwd=True,mask=False): #rotate pi/4 clockwise
    return aug_flipH(aug_T(x,fwd,mask),fwd,mask) if fwd else \
        aug_T(aug_flipH(x,fwd,mask),fwd,mask)

def aug_rot_2T(x,fwd=True,mask=False): #transpose and rotate pi/2
    return aug_rot_2(aug_T(x,fwd,mask),fwd,mask)

import torchvision

def color_jitter(x, fwd=True, mask=False):
    if mask:
        return x
    else:
        return some_transform(x)


class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  #add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x

rl = RandomLighting(0.05, 0.05)




trms_side_on = [aug_unit,aug_flipH]
trms_top_down = [aug_unit,aug_flipV]
trms_dihedral = [aug_unit,aug_flipH,aug_flipV,aug_T,aug_rot_2,aug_rot_2T,
                 aug_rot_4cw,aug_rot_4cr]

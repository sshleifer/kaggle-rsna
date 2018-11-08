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

trms_side_on = [aug_unit,aug_flipH]
trms_top_down = [aug_unit,aug_flipV]
trms_dihedral = [aug_unit,aug_flipH,aug_flipV,aug_T,aug_rot_2,aug_rot_2T,
                 aug_rot_4cw,aug_rot_4cr]

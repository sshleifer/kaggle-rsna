# the actual model function is defined here
# NOTE: the comments are meant to help understand/check the input-output sizes of the tensor
#       and they assume an input image size of 256x256,
#       but the model can accept larger/smaller sizes (tho I haven't debugged it yet)

# Define the nn convolutional block
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv
from torch.autograd import Variable

class conv_block(nn.Module):
    """
    Define the [convolution - batch normalization - activation] block
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 bn_momentum=0.9, alpha_leaky=0.03):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x):
        return self.activ(self.bn(self.conv(x)))


# Define the nn transposed convolutional block
class conv_t_block(nn.Module):
    """
    Define the [convolution_transpose - batch normalization - activation] block
    """

    def __init__(self, in_channels, out_channels, output_size=None, kernel_size=3, bias=True,
                 bn_momentum=0.9, alpha_leaky=0.03):
        super(conv_t_block, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2,
                                         padding=1,
                                         bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x, output_size):
        return self.activ(self.bn(self.conv_t(x, output_size=output_size)))

class PneumoniaUNET(nn.Module):
    def __init__(self):
        super(PneumoniaUNET, self).__init__()

        self.down_1 = conv_block(in_channels=1, out_channels=64)
        self.down_2 = conv_block(in_channels=64, out_channels=128)
        self.down_3 = conv_block(in_channels=128, out_channels=256)
        self.down_4 = conv_block(in_channels=256, out_channels=512)
        self.down_5 = conv_block(in_channels=512, out_channels=512)

        self.middle = conv_block(in_channels=512, out_channels=512)
        self.middle_t = conv_t_block(in_channels=512, out_channels=256)

        self.up_5 = conv_block(in_channels=768, out_channels=512)
        self.up_5_t = conv_t_block(in_channels=512, out_channels=256)
        self.up_4 = conv_block(in_channels=768, out_channels=512)
        self.up_4_t = conv_t_block(in_channels=512, out_channels=128)
        self.up_3 = conv_block(in_channels=384, out_channels=256)
        self.up_3_t = conv_t_block(in_channels=256, out_channels=64)
        self.up_2 = conv_block(in_channels=192, out_channels=128)
        self.up_2_t = conv_t_block(in_channels=128, out_channels=32)
        self.up_1 = conv_block(in_channels=97, out_channels=1)

    def forward(self, x, feat_batch):
        down1 = self.down_1(x)  # (1x256x256 -> 64x256x256)
        out = F.max_pool2d(down1, kernel_size=2, stride=2)  # (64x256x256 -> 64x128x128)

        down2 = self.down_2(out)  # (64x128x128 -> 128x128x128)
        out = F.max_pool2d(down2, kernel_size=2, stride=2)  # (128x128x128 -> 128x64x64)

        down3 = self.down_3(out)  # (128x64x64 -> 256x64x64)
        out = F.max_pool2d(down3, kernel_size=2, stride=2)  # (256x64x64 -> 256x32x32)

        down4 = self.down_4(out)  # (256x32x32 -> 512x32x32)
        out = F.max_pool2d(down4, kernel_size=2, stride=2)  # (512x32x32 -> 512x16x16)

        down5 = self.down_5(out)  # (512x16x16 -> 512x16x16)
        out = F.max_pool2d(down5, kernel_size=2, stride=2)  # (512x16x16 -> 512x8x8)

        out = self.middle(out)  # (512x8x8 -> 512x8x8)
        out = self.middle_t(out, output_size=down5.size())  # (512x8x8 -> 256x16x16)

        out = torch.cat([down5, out], 1)  # (512x16x16-concat-256x16x16 -> 768x16x16)
        out = self.up_5(out)  # (768x16x16 -> 512x16x16)
        out = self.up_5_t(out, output_size=down4.size())  # (512x16x16 -> 256x32x32)

        out = torch.cat([down4, out], 1)  # (512x32x32-concat-256x32x32 -> 768x32x32)
        out = self.up_4(out)  # (768x32x32 -> 512x32x32)
        out = self.up_4_t(out, output_size=down3.size())  # (512x32x32 -> 128x64x64)

        out = torch.cat([down3, out], 1)  # (256x64x64-concat-128x64x64 -> 384x64x64)
        out = self.up_3(out)  # (384x64x64 -> 256x64x64)
        out = self.up_3_t(out, output_size=down2.size())  # (256x64x64 -> 64x128x128)

        out = torch.cat([down2, out], 1)  # (128x128x128-concat-64x128x128 -> 192x128x128)
        out = self.up_2(out)  # (192x128x128 -> 128x128x128)
        out = self.up_2_t(out, output_size=down1.size())  # (128x128x128 -> 32x256x256)

        out = torch.cat([down1, out, feat_batch], 1)  # (64x256x256-concat-32x256x256-concat1x256x256 -> 97x256x256)
        out = self.up_1(out)  # (97x256x256 -> 1x256x256)

        return out

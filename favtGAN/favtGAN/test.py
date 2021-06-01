import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys


import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

#from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch


"""
Args:

Please pass:

--dataset_name eurecom_adas_iris

--experiment exp_b2

--labels_csv eurecom_adas_iris_d # do not add the '.csv' extension

--long_dir use the directory you created in images/test_results specific to this '2000 epochs' run

"""

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
parser.add_argument("--labels_csv", type=str, default="none", help="name of the labels csv file (no file ext)")
parser.add_argument("--long_dir", type=str, default="none", help="test images saved to long 2000 directory")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# ===========================================================
# i copied from models.py directly here
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        print("UNetDown x.shape", x.size())
        # print("UNetDown x is:", x)
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print("UNetUp x input to forward is:", x.size())
        x = self.model(x)
        # print("UnetUp x = self.model(x) is:", x.size())
        x = torch.cat((x, skip_input), 1)
        # print("UNetUp x after torch.cat:", x.size())

        return x


class GeneratorUNet(nn.Module):
    # def __init__(self, in_channels=3, out_channels=3):
    #def __init__(self, latent_dim, img_shape):
    def __init__(self, img_shape):
        super(GeneratorUNet, self).__init__()

        # self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        # try the below from bicyclegan

        channels, self.h, self.w = img_shape

        # this is to make the mul math work out
        # so that [batch x 1] x [1, 256*256]  = [batch, 256*256] => .view as 4D

        allowable_labels_per_batch = 1
        self.fc = nn.Linear(allowable_labels_per_batch, self.h * self.w)

        # self.down1 = UNetDown(in_channels, 64, normalize=False)
        # bicyclegan adds 1 to the channel

        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, labels):
        # U-Net generator with skip connections from encoder to decoder

        print("x shape into gen is:", x.size())
        print("labels into gen is:", labels.size())

        labels = self.fc(labels).view(labels.size(0), 1, self.h, self.w)


        d1 = self.down1(torch.cat((x, labels), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        # print("self.final(u7).shape:", self.final(u7).size())
        return self.final(u7)


# Initialize generator and discriminator
input_shape = (opt.channels, opt.img_height, opt.img_width)

generator = GeneratorUNet(input_shape)

if cuda:
    generator = generator.cuda()

# ====== Load pretrained models ======

generator.load_state_dict(torch.load("experiments/pix2pix_mods_label_loss/saved_models/%s/generator_1990.pth" % (opt.experiment)))


# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    TestImageDataset(root = "experiments/data/%s" % opt.dataset_name,
        transforms_=transforms_,
        annots_csv = "experiments/data/labels/{}.csv".format(opt.labels_csv),
        mode="test"),
    batch_size=1,
    shuffle=False, # this way I can cross-ref the file names with the images
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ===== Generate Predictions ====

for i, batch in enumerate(test_dataloader):
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))
    labels = Variable(batch["LAB"].type(FloatTensor))
    fake_B = generator(real_A, labels) #this is the output we're looking for
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "experiments/pix2pix_mods_label_loss/images/test_results/%s/%s.png" % (opt.long_dir, i), nrow=5, normalize=True)

import argparse
import os
import numpy as np
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch

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
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
parser.add_argument("--gpu_num", type=int, default=0, help="gpu card to use for training")
parser.add_argument("--out_file", type=str, default="out", help="name of output log files")
parser.add_argument("--annots_csv", type=str, default="none", help="csv file path for labels")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
opt = parser.parse_args()
print(opt)

os.makedirs("favtGAN/images/%s" % opt.experiment, exist_ok=True)
os.makedirs("favtGAN/saved_models/%s" % opt.experiment, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.gpu_num)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
auxiliary_loss = torch.nn.CrossEntropyLoss()
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

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
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):

    def __init__(self, img_shape):
        super(GeneratorUNet, self).__init__()

        channels, self.h, self.w = img_shape
        allowable_labels_per_batch = 1
        self.fc = nn.Linear(allowable_labels_per_batch, self.h * self.w)

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
        return self.final(u7)

##############################
#        Discriminator
##############################

class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        channels, self.h, self.w = img_shape

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block((channels * 2), 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
        self.aux_layer = nn.Sequential(nn.Linear((channels * 2) * self.h * self.w, opt.n_classes), nn.Softmax())

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        d_in = img_input
        output = self.model(d_in)
        out = d_in.view(d_in.shape[0], -1)
        label = self.aux_layer(out)
        return output, label
# ===========================================================
# Initialize generator and discriminator
input_shape = (opt.channels, opt.img_height, opt.img_width)
generator = GeneratorUNet(input_shape)
discriminator = Discriminator(input_shape)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    auxiliary_loss.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("favtGAN/saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator.load_state_dict(torch.load("favtGAN/saved_models/%s/discriminator_%d.pth" % (opt.experiment, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(root = "experiments/data/%s" % opt.dataset_name,
        annots_csv = opt.annots_csv,
        transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True,
)

test_dataloader = DataLoader(
    ImageDataset(root = "experiments/data/%s" % opt.dataset_name,
        transforms_=transforms_,
        annots_csv = opt.annots_csv,
        mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Generates a sample test image
def sample_images(batches_done):
    """Saves a generated sample from the validation set
    currently set to go from A to B; visible to thermal """
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    labels = Variable(imgs["LAB"].type(FloatTensor))
    fake_B = generator(real_A, labels)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.experiment, batches_done), nrow=5, normalize=True)

# ----------
#  Training
# ----------

prev_time = time.time()
f = open('favtGAN/{}.txt'.format(opt.out_file), 'a+')

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        labels = Variable(batch["LAB"].type(FloatTensor))

        # Adversarial ground truths
        valid_ones = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        # One-side label smoothing per Salimans, fill with 0.90 on valid/real only
        valid = valid_ones.fill_(0.9)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        print("+ + + optimizer_G.zero_grad() + + +")
        optimizer_G.zero_grad()

        allowable_labels_per_batch = 1
        gen_labels = Variable(FloatTensor(np.random.randint(0, opt.n_classes, (opt.batch_size, allowable_labels_per_batch))))

        # Generate
        fake_B = generator(real_A, gen_labels)
        pred_fake, pred_label = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)

        if opt.batch_size > 1:
            gen_labels = gen_labels.type(torch.LongTensor)
            gen_labels = gen_labels.squeeze_()
            gen_labels = gen_labels.to(device='cuda')
        elif opt.batch_size ==1:
            gen_labels = gen_labels.type(torch.LongTensor)
            gen_labels = gen_labels.squeeze_(dim=0) # batch size must be torch.size[1] by 0 dim
            gen_labels = gen_labels.to(device='cuda')

        # Aux loss
        label_loss = auxiliary_loss(pred_label, gen_labels)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # Total loss
        loss_G = 0.5 * (loss_GAN + label_loss + lambda_pixel * loss_pixel)

        loss_G.backward()
        optimizer_G.step()
        print("+ + + optimizer_G.step() + + +")

        # ---------------------
        #  Train Discriminator
        # ---------------------

        print("+ + + optimizer_D.zero_grad() + + +")
        optimizer_D.zero_grad()

        # Real
        pred_real, real_aux = discriminator(real_B, real_A)
        # Adv real d
        loss_real = criterion_GAN(pred_real, valid)

        if opt.batch_size > 1:
            labels = labels.type(torch.LongTensor)
            labels = labels.squeeze_()
            labels = labels.to(device='cuda')
        elif opt.batch_size ==1:
            labels = labels.type(torch.LongTensor)
            labels = labels.squeeze_(dim=0) # batch size must be torch.size[1] by 0 dim
            labels = labels.to(device='cuda')
        # Aux real D
        real_label_loss = auxiliary_loss(real_aux, labels)
        # Total real D
        d_real = loss_real + real_label_loss  # image + label loss

        # Fake
        gen_labels = gen_labels.type(torch.FloatTensor) # need to return back to float for the Discriminator
        gen_labels = torch.unsqueeze(gen_labels, 1) # return it back to its 2D tensor
        gen_labels = gen_labels.to(device='cuda')

        pred_fake, fake_aux = discriminator(fake_B.detach(), real_A)
        # Adv fake D
        loss_fake = criterion_GAN(pred_fake, fake)

        # you always need to turn the criterion(input, target) where input is 2D and target is 1D
        if opt.batch_size > 1:
            gen_labels = gen_labels.type(torch.LongTensor)
            gen_labels = gen_labels.squeeze_()
            gen_labels = gen_labels.to(device='cuda')
            print("generated label is after squeeze():", gen_labels.size())
        elif opt.batch_size ==1:
            gen_labels = gen_labels.type(torch.LongTensor)
            gen_labels = gen_labels.squeeze_(dim=0) # batch size must be torch.size[1] by 0 dim
            gen_labels = gen_labels.to(device='cuda')
            print("generated label is after squeeze():", gen_labels.size())

        # Aux fake D
        fake_label_loss = auxiliary_loss(fake_aux, gen_labels) # image + label loss
        # Total fake D
        d_fake = loss_fake + fake_label_loss
        # Total discriminator loss
        loss_D = 0.5 * (d_real + d_fake)

        loss_D.backward()
        optimizer_D.step()
        print("+ + + optimizer_D.step() + + +")

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "favtGAN/saved_models/%s/generator_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator.state_dict(), "favtGAN/saved_models/%s/discriminator_%d.pth" % (opt.experiment, epoch))

f.close()

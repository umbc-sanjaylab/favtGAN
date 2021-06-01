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

# from models import *
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
#parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
parser.add_argument("--gpu_num", type=int, default=0, help="gpu card to use for training")
parser.add_argument("--out_file", type=str, default="out", help="name of output log files")
parser.add_argument("--annots_csv", type=str, default="none", help="csv file path for labels")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
parser.add_argument("--lambda_adv", type=float, default=0.2, help="adversarial scaling factor")
opt = parser.parse_args()
print(opt)

os.makedirs("experiments/pix2pix_mods_label_loss/images/%s" % opt.experiment, exist_ok=True)
os.makedirs("experiments/pix2pix_mods_label_loss/saved_models/%s" % opt.experiment, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.gpu_num)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100
lambda_adv = opt.lambda_adv

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
print("patch:", patch)


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

        """
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )
        """
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

        # gen_input = torch.cat((self.label_emb(labels), x), -1)
        # gen_input = torch.cat((x.view(x.size(0), -1), self.label_emb(labels)), -1)
        # print("gen_input.shape after torch.cat", gen_input.size())
        # print("gen_input output after torch.cat:", gen_input)

        # below is what makes it 4D for UNET
        labels = self.fc(labels).view(labels.size(0), 1, self.h, self.w)
        # print("labels after self.fc(labels) is:", labels.size())

        d1 = self.down1(torch.cat((x, labels), 1))
        # d1 = torch.cat((x.view(x.size(0), -1), self.label_emb(labels)), -1)
       #  print("UNETDown first layer is d1.size:", d1.size())

        # d1 = self.down1(gen_input)
        # d1 = self.down1(x)
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


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    #def __init__(self, img_shape, in_channels=3):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        channels, self.h, self.w = img_shape
        print("channels to D:", channels)  # 3
        print("self.h to D:", self.h)  # 256
        print("self.w to D:", self.w)  # 256

        # self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        allowable_labels_per_batch = 1
        self.fc = nn.Linear(allowable_labels_per_batch, self.h * self.w)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]

            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # *discriminator_block(in_channels * 2, 64, normalization=False),
            # *discriminator_block((channels + 1) * 2, 64, normalization=False),
            *discriminator_block((channels * 2) + 1, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

        # The height and width of downsampled image
        #img_size = self.h
        #ds_size = img_size // 2 ** 4
        # The label loss - I don't know if this is right?
        self.aux_layer = nn.Sequential(nn.Linear(((channels * 2) + 1) * self.h * self.w, opt.n_classes), nn.Softmax())


    # def forward(self, img_A, img_B):
    def forward(self, img_A, img_B, labels):
        # Concatenate image and condition image by channels to produce input

        # images
        img_input = torch.cat((img_A, img_B), 1)
        print("img_input to forward D:", img_input.size()) # torch.Size([2, 6, 256, 256])

        # labels
        # here, we flatten it into a fc and reshape to 4D
        labels = self.fc(labels).view(labels.size(0), 1, self.h, self.w)
        print("labels after reshaping for D:", labels.size())

        # concat images and labels
        # d_in = torch.cat((img_input.view(img_input.size(0), -1), self.label_emb(labels)), -1)
        d_in = torch.cat((img_input, labels), 1)
        print("d_in.shape:", d_in.shape)  # must return 4d
        print("self.model(d_in) prediction:", self.model(d_in).size())
        #return self.model(d_in)

        output = self.model(d_in)
        print("discriminator output:", output.size())

        out = d_in.view(d_in.shape[0], -1) #batch_size, cols
        print("out for labels aux layer:", out.size())
        label = self.aux_layer(out)
        print("discriminator label which should match predicted label:", label.size())  # torch.Size([1, 2])
        # print("===next===")

        return output, label



# ===========================================================
# Initialize generator and discriminator
input_shape = (opt.channels, opt.img_height, opt.img_width)

# generator = GeneratorUNet()
#generator = GeneratorUNet(opt.latent_dim, input_shape)
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
    generator.load_state_dict(torch.load("experiments/pix2pix_mods_label_loss/saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator.load_state_dict(torch.load("experiments/pix2pix_mods_label_loss/saved_models/%s/discriminator_%d.pth" % (opt.experiment, opt.epoch)))
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

# Note - ImageDataset will automatically infer the image directory structure
# and labels should be in the right structure: foo/train/facades/ 01.png, foo/train/maps/01.png

# ImageDataset comes from datasets.py

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

# this generates output after training has nothing to do with dataloader

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

f = open('experiments/pix2pix_mods_label_loss/{}.txt'.format(opt.out_file), 'a+') # Open for reading and writing.  The file is created if it does not exist


for epoch in range(opt.epoch, opt.n_epochs):
    # for i, batch in enumerate(dataloader):

    for i, batch in enumerate(dataloader):

        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        labels = Variable(batch["LAB"].type(FloatTensor))

        print("real_A from enumerate(dataloader):", real_A.size())
        print("real_B from enumerate(dataloader):", real_B.size())
        print("labels from enumerate(dataloader):", labels.size())
        print("label here:", labels)

        # Adversarial ground truths
        valid_ones = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        # One-side label smoothing per Salimans, fill with 0.90 on valid/real only
        valid = valid_ones.fill_(0.9)

        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
        # print(fake)


        # ------------------
        #  Train Generators
        # ------------------

        print("+ + + optimizer_G.zero_grad() + + +")

        optimizer_G.zero_grad()

        # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.batch_size))) # (random ints 0 to 2, batch size of 1)
        allowable_labels_per_batch = 1

        #noise, c_f
        gen_labels = Variable(FloatTensor(np.random.randint(0, opt.n_classes, (opt.batch_size, allowable_labels_per_batch))))

        print("original labels: \n", gen_labels)
        print("gen labels.size:", gen_labels.size())
        print("gen labels.size(0):", gen_labels.size(0))
        print(gen_labels)

        # Generate
        print("real_A shape which will input into generator forward as x is:", real_A.size())
        #print("====generator=====\n")

        #V3 uses real labels not gen labels
        #fake_B = generator(real_A, gen_labels)
        fake_B = generator(real_A, labels)

        #V3 uses real labels not gen labels
        #print("size of gen_labels before discriminator(fake_B, real_A, gen_labels):", gen_labels.size())
        #pred_fake, pred_label = discriminator(fake_B, real_A, gen_labels)

        print("size of real labels before discriminator(fake_B, real_A, labels):", labels.size())
        pred_fake, pred_label = discriminator(fake_B, real_A, labels)

        print("pred_label:", pred_label.size())
        print("real label from data is:", labels.size())

        print("====G losses calculating=========\n")
        # adversarial G loss
        loss_GAN = criterion_GAN(pred_fake, valid)
        print("loss_GAN:", loss_GAN)

        # Reshape for cross entropy loss function
        # V3 - criterion(input, target) - how well predicted label matches real labels

        if opt.batch_size > 1:
            labels = labels.type(torch.LongTensor)
            labels = labels.squeeze_()
            labels = labels.to(device='cuda')
            print("label is after squeeze():", labels.size())
        elif opt.batch_size ==1:
            labels = labels.type(torch.LongTensor)
            labels = labels.squeeze_(dim=0) # batch size must be torch.size[1] by 0 dim
            labels = labels.to(device='cuda')
            print("label is after squeeze():", labels.size())

        #V3 we use real labels
        #label_loss = auxiliary_loss(pred_label, gen_labels)
        label_loss = auxiliary_loss(pred_label, labels)
        print("label_loss:", label_loss)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        print("loss_pixel:", loss_pixel)

        # Total loss
        #loss_G = loss_GAN + lambda_pixel * loss_pixel
        loss_G = lambda_adv*loss_GAN + (1 - lambda_adv)*label_loss + lambda_pixel*loss_pixel
        print("====TOTAL G:", loss_G)

        loss_G.backward()

        optimizer_G.step()
        print("+ + + optimizer_G.step() + + +")

        # ---------------------
        #  Train Discriminator
        # ---------------------

        print("+ + + optimizer_D.zero_grad() + + +")

        optimizer_D.zero_grad()

        # Real loss:
        labels = labels.type(torch.FloatTensor) # need to return back to float for the Discriminator
        labels = torch.unsqueeze(labels, 1) # return it back to its 2D tensor
        labels = labels.to(device='cuda')
        print("real labels going into discriminator(real_B, real_A, labels)", labels)
        print("size:", labels.size())

        pred_real, real_aux = discriminator(real_B, real_A, labels)

        # D_loss adversarial loss
        loss_real = criterion_GAN(pred_real, valid)

        if opt.batch_size > 1:
            labels = labels.type(torch.LongTensor)
            labels = labels.squeeze_()
            labels = labels.to(device='cuda')
            print("label is after squeeze():", labels.size())
        elif opt.batch_size ==1:
            labels = labels.type(torch.LongTensor)
            labels = labels.squeeze_(dim=0) # batch size must be torch.size[1] by 0 dim
            labels = labels.to(device='cuda')
            print("label is after squeeze():", labels.size())

        real_label_loss = auxiliary_loss(real_aux, labels)

        # add the losses
        #V3 uses a scaling factor since loss_real is MSE and real_label_loss is LOG LOSS

        d_real = lambda_adv*loss_real + (1 - lambda_adv)*real_label_loss

        print("=====d_real:", d_real)

        # Fake loss:
        gen_labels = gen_labels.type(torch.FloatTensor) # need to return back to float for the Discriminator
        gen_labels = torch.unsqueeze(gen_labels, 1) # return it back to its 2D tensor
        gen_labels = gen_labels.to(device='cuda')

        #print("gen labels going into discriminator(fake_B.detach(), real_A, gen_labels)", gen_labels)
        #print("size:", gen_labels.size())

        pred_fake, fake_aux = discriminator(fake_B.detach(), real_A, gen_labels)

        # D_fake adversarial loss
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

        fake_label_loss = auxiliary_loss(fake_aux, gen_labels)

        #add the losses
        #Again, V3 scales the adversarial loss

        d_fake = lambda_adv*loss_fake + (1-lambda_adv)*fake_label_loss
        print("====d_fake", d_fake)


        #====== Total loss ====
        loss_D = 0.5 * (d_real + d_fake)
        print("====TOTAL D loss:", loss_D)

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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D_real adv: %f, aux: %f, total: %f] [D_fake adv: %f, aux: %f, total: %f][G loss: %f, pixel: %f, adv: %f, aux: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_real.item(), #D_real adv %f
                real_label_loss.item(), #D_real_aux %f
                d_real.item(), # D_real total %f
                loss_fake.item(), #D_fake adv %f
                fake_label_loss.item(), #D_fake_aux %f
                d_fake.item(), #D_fake total %f
                loss_G.item(), #%f - total G loss
                loss_pixel.item(), #%f
                loss_GAN.item(), #%f - adv G loss
                label_loss.item(), # G aux loss %f
                time_left, #%s
            )
        )

        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D_real adv: %f, aux: %f, total: %f] [D_fake adv: %f, aux: %f, total: %f][G loss: %f, pixel: %f, adv: %f, aux: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_real.item(), #D_real adv %f
                real_label_loss.item(), #D_real_aux %f
                d_real.item(), # D_real total %f
                loss_fake.item(), #D_fake adv %f
                fake_label_loss.item(), #D_fake_aux %f
                d_fake.item(), #D_fake total %f
                loss_G.item(), #%f - total G loss
                loss_pixel.item(), #%f
                loss_GAN.item(), #%f - adv G loss
                label_loss.item(), # G aux loss %f
                time_left, #%s
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "experiments/pix2pix_mods_label_loss/saved_models/%s/generator_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator.state_dict(), "experiments/pix2pix_mods_label_loss/saved_models/%s/discriminator_%d.pth" % (opt.experiment, epoch))

f.close()

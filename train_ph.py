import os
from os import makedirs

import yaml
import random

from torchvision import transforms

import model
import numpy as np
from glob import glob
from easydict import EasyDict
import hashlib
from PIL import Image, ImageOps
from torch import optim

import utils
import phash
from dataset import MyData, train_test_dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lpips


def discriminator_loss(discriminator, fake_images, real_images):
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images)

    if isinstance(real_logits, tuple):
        real_logits = real_logits[0]
    if isinstance(fake_logits, tuple):
        fake_logits = fake_logits[0]

    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    )
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    return real_loss + fake_loss



with open('cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

if not os.path.exists(args.saved_models):
    os.makedirs(args.saved_models)

def main():
    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)

    # import dataset, resize image to 400x400 and compute perceptual hash
    dataset = MyData(args.train_path, args.secret_size, size=(400, 400))
    datasets = train_test_dataset(dataset, test_split=0.1)
    train_val_dataset = datasets['train']
    subsets = train_test_dataset(train_val_dataset, test_split=0.02)
    train_dataset = subsets['train']
    val_dataset   = subsets['test']
    test_dataset  = datasets['test']
    # add dataset to tensorboard
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True,  pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)

    encoder = model.Encoder1()
    decoder = model.Decoder1(secret_size=64)
    discriminator = model.Discriminator()
    lpips_alex = lpips.LPIPS(net="vgg", verbose=False)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()
        lpips_alex.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimize_loss, gamma=0.99)


    height = 400
    width = 400

    total_steps = len(train_dataset) // args.batch_size + 1
    print(f"Total steps: {total_steps}")
    global_step = 0


    

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            image_input, secret_input = next(iter(train_loader))
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()

        ## Encoder Process
        print("start encoder process at step", global_step)
        residual_phash = encoder((secret_input, image_input))
        image_firststage = residual_phash + image_input

        writer.add_image("encode/CoverImage(I)", image_input[0], global_step)
        writer.add_image("encode/ResidualPHash(R1)", residual_phash[0], global_step)
        writer.add_image("encode/FirstStage(W1)", image_firststage[0], global_step)

        ## Random transform/ attack the image

        image_transformed = model.transform_net(image_firststage, args, global_step)
        image_transformed_pil = [transforms.ToPILImage()(img.squeeze().detach().cpu()) for img in image_transformed]

        ## Decoder Process
        print("start decoder process at step", global_step)
        # extracted perceptual hash ph
        extracted_secret = decoder(image_transformed)
        phash_decoded_hash = phash.calculate_phash(image_transformed_pil, hash_size=int(args.secret_size ** 0.5))
        # print(type(phash_decoded_hash))
        phash_decoded = np.array([
            [int(bit) for bit in bin(int(hash_str, 16))[2:].zfill(args.secret_size)]
            for hash_str in phash_decoded_hash
        ])
        # print(type(phash_decoded))
        phash_decoded = torch.from_numpy(phash_decoded).float()
        # print(type(phash_decoded))
        if args.cuda:
            phash_decoded = phash_decoded.cuda()
        phash_bit_acc, phash_str_acc = model.get_secret_acc(extracted_secret, phash_decoded)
        writer.add_scalar("acc/ExtractedPHashBitAcc", phash_bit_acc, global_step)
        writer.add_scalar("acc/ExtractedPHashStrAcc", phash_str_acc, global_step)

        bit_acc, str_acc = model.get_secret_acc(secret_input, phash_decoded)
        writer.add_scalar("acc/SecretBitAcc", bit_acc, global_step)
        writer.add_scalar("acc/SecretStrAcc", str_acc, global_step)

        # Loss Calculation
        print("start loss calculation at step", global_step)

        rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
        rnd_tran = np.random.uniform() * rnd_tran 
        Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
        if args.cuda:
            Ms = Ms.cuda()

        # LE1 = LR(E1) + LP(W1, I) + LC(I, W1).
        lpips_loss_encoder = (lpips_alex(image_input, image_firststage).mean() + lpips_alex(image_firststage, image_input).mean())/2
        l2_loss_encoder = (torch.nn.functional.mse_loss(image_input, image_firststage) + torch.nn.functional.mse_loss(image_firststage, image_input))/2
        lc_loss_encoder = discriminator_loss(discriminator, image_firststage, image_input)


        # LD1 = LBCE(D1(W2), ph)
        bce_loss_decoder = (torch.nn.functional.binary_cross_entropy_with_logits(extracted_secret, secret_input)+torch.nn.functional.binary_cross_entropy_with_logits(secret_input, extracted_secret))/2

        # Weighting the losses
        w_l2_encoder = 1.0
        w_lpips_encoder = 0.5
        w_bce_decoder = 1.0
        w_discriminator = 0.1

        # LE1 = LR(E1) + LP(W1, I) + LC(I, W1).
        loss_encoder = w_l2_encoder * l2_loss_encoder + w_lpips_encoder * lpips_loss_encoder + lc_loss_encoder * w_discriminator
        # LD1 = LBCE(D1(W2), ph)
        loss_decoder = w_bce_decoder * bce_loss_decoder

        total_loss = loss_encoder + loss_decoder

        if global_step < args.no_im_loss_steps:
            optimize_secret_loss.zero_grad()
            bce_loss_decoder.backward()
            optimize_secret_loss.step()
        else:
            optimize_loss.zero_grad()
            total_loss.backward()
            optimize_loss.step()
            if not args.no_gan:
                optimize_dis.zero_grad()
                lc_loss_encoder.backward()
                optimize_dis.step()

        writer.add_scalar("loss/Total_Loss", total_loss.item(), global_step)
        writer.add_scalar("loss/LE1_Loss_Encoder", loss_encoder.item(), global_step)
        writer.add_scalar("loss/LD1_Loss_Decoder", loss_decoder.item(), global_step)
        writer.add_scalar("loss/Learning_Rate", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("loss/Random_Transform", rnd_tran, global_step)

        global_step += 1
        if global_step >= args.num_steps:
            break
        if global_step == 1:
            print("Started training...")
        if global_step % 10 == 0:
            print(f"Step {global_step}, Total Loss: {total_loss.item():.4f}")
        if global_step % 30 == 0:
            torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
            torch.save(decoder, os.path.join(args.saved_models, "decoder.pth"))

        scheduler.step()

    writer.close()
    torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
    torch.save(decoder, os.path.join(args.saved_models, "decoder.pth"))


if __name__ == '__main__':
    main()

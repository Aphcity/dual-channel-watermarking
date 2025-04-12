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
from dataset import MyData
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
    # add dataset to tensorboard
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    encoder = model.Encoder1()
    encoder1= model.Encoder2()
    decoder = model.Decoder1(secret_size=64)
    decoder1=model.Decoder2()
    decoder2=model.Decoder3(secret_size=32)
    discriminator = model.Discriminator()
    lpips_alex = lpips.LPIPS(net="vgg", verbose=False)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
        decoder2 = decoder2.cuda()
        discriminator = discriminator.cuda()
        lpips_alex.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()},
              {'params': encoder1.parameters()},
              {'params': decoder1.parameters()},
              {'params': decoder2.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimize_loss, gamma=0.999875)


    height = 400
    width = 400

    total_steps = len(dataset) // args.batch_size + 1
    print(f"Total steps: {total_steps}")
    global_step = 0


    

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            image_input, secret_input = next(iter(dataloader))
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
        encoder_output = encoder((secret_input, image_input))  
        encoded_image = encoder_output

        writer.add_image("encode/CoverImage(I)", image_input[0], global_step)
        # writer.add_image("SecretImage(FC1)", secret_input[0], global_step)
        writer.add_image("encode/FirstStageWateredImage(W1)", encoded_image[0], global_step)

        autoencoder = model.Autoencoder(latent_dim=2048)
        if args.cuda:
            autoencoder = autoencoder.cuda()
        latent= autoencoder.encoder(encoded_image)

        latent_np = latent.detach().cpu().numpy()
        sha3_hash_list = []
        for i in range(latent_np.shape[0]): 
            single_latent = latent_np[i]  
            sha3_hash = hashlib.sha3_256(single_latent.tobytes()).digest()  
            sha3_hash_tensor = torch.tensor(list(sha3_hash), dtype=torch.float32).to(image_input.device)  # Convert to tensor
            sha3_hash_list.append(sha3_hash_tensor)  
        sha3_hash_tensor_batch = torch.stack(sha3_hash_list)
        encoder1_output = encoder1((sha3_hash_tensor_batch, encoded_image))
        writer.add_image("encode/ResidualImage(R)", encoder1_output[0], global_step)
        final_encoded_image = encoded_image + encoder1_output
        writer.add_image("encode/SecondStageWateredImage(W2)", final_encoded_image[0], global_step)

        ## Decoder Process

        extracted_secret = decoder(final_encoded_image)  

        reconstructed_residual = decoder1(final_encoded_image) 
        reconstructed_image = final_encoded_image - reconstructed_residual
        writer.add_image("decode/ResidualImage(R')", reconstructed_residual[0], global_step)
        writer.add_image("decode/WateredImage(W1)", reconstructed_image[0], global_step)

        extracted_hash = decoder2(reconstructed_residual)

        latent_decoder = autoencoder.decoder(reconstructed_image)
        latent_decoder_np = latent_decoder.detach().cpu().numpy()
        sha3_hash_list_decoder = []
        for i in range(latent_decoder_np.shape[0]):
            single_latent_decoder = latent_decoder_np[i]
            sha3_hash_decoder = hashlib.sha3_256(single_latent_decoder.tobytes()).digest()
            sha3_hash_tensor_decoder = torch.tensor(list(sha3_hash_decoder), dtype=torch.float32).to(image_input.device)
            sha3_hash_list_decoder.append(sha3_hash_tensor_decoder)
        sha3_hash_tensor_batch_decoder = torch.stack(sha3_hash_list_decoder)

        # Loss Calculation

        rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
        rnd_tran = np.random.uniform() * rnd_tran 
        Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
        if args.cuda:
            Ms = Ms.cuda()

        # LE1 = LR(E1) + LP(W1, I) + LC(I, W1).
        lpips_loss_encoder = (lpips_alex(image_input, encoded_image).mean() + lpips_alex(encoded_image, image_input).mean())/2
        l2_loss_encoder = (torch.nn.functional.mse_loss(image_input, encoded_image) + torch.nn.functional.mse_loss(encoded_image, image_input))/2

        # LE2 = LR(E2) + LP(W2, I) + LC(I, W2).
        lpips_loss_encoder1 = (lpips_alex(image_input, final_encoded_image).mean() + lpips_alex(final_encoded_image, image_input).mean())/2
        l2_loss_encoder1 = (torch.nn.functional.mse_loss(image_input, final_encoded_image) + torch.nn.functional.mse_loss(final_encoded_image, image_input))/2
        
        # LD1 = LBCE(D1(W2), ph)
        bce_loss_decoder = torch.nn.functional.binary_cross_entropy_with_logits(extracted_secret, secret_input)

        # LD2 = L1(R,R') + LP (R,R') + MSE(R,R')
        l1_loss_decoder1 = torch.nn.functional.l1_loss(reconstructed_residual, encoder1_output)
        l2_loss_decoder1 = torch.nn.functional.mse_loss(reconstructed_residual, encoder1_output)
        lpips_loss_decoder1 = lpips_alex(reconstructed_residual, encoder1_output).mean()
        
        D_loss = discriminator_loss(discriminator, final_encoded_image, image_input)
        print(secret_input.shape)

        # LD3 = LBCE(D3(R'), h).
        bce_loss_decoder2 = torch.nn.functional.binary_cross_entropy_with_logits(extracted_hash, sha3_hash_tensor_batch)


        # Weighting the losses
        w_l2_encoder = 1.0  
        w_lpips_encoder = 0.5  
        w_l2_encoder1 = 1.0 
        w_lpips_encoder1 = 0.5  
        w_bce_decoder = 1.0  
        w_l1_decoder1 = 0.8  
        w_l2_decoder1 = 0.8 
        w_lpips_decoder1 = 0.5  
        w_bce_decoder2 = 1.0  
        w_discriminator = 0.1

        total_loss = (
            w_l2_encoder * l2_loss_encoder +
            w_lpips_encoder * lpips_loss_encoder +
            w_l2_encoder1 * l2_loss_encoder1 +
            w_lpips_encoder1 * lpips_loss_encoder1 +
            w_bce_decoder * bce_loss_decoder +
            w_l1_decoder1 * l1_loss_decoder1 +
            w_l2_decoder1 * l2_loss_decoder1 +
            w_lpips_decoder1 * lpips_loss_decoder1 +
            w_bce_decoder2 * bce_loss_decoder2 +
            w_discriminator * D_loss
                )

        optimize_loss.zero_grad()
        total_loss.backward()
        optimize_loss.step()

        if not args.no_gan:
            optimize_dis.zero_grad()
            D_loss.backward()
            optimize_dis.step()

        writer.add_scalar("loss/Total_Loss", total_loss.item(), global_step)
        # writer.add_scalar("loss/L2_Loss_Encoder", l2_loss_encoder.item(), global_step)
        # writer.add_scalar("loss/L2_Loss_Encoder1", l2_loss_encoder1.item(), global_step)
        # writer.add_scalar("loss/LPIPS_Loss_Encoder", lpips_loss_encoder.item(), global_step)
        # writer.add_scalar("loss/LPIPS_Loss_Encoder1", lpips_loss_encoder1.item(), global_step)
        # writer.add_scalar("loss/BCE_Loss_Decoder", bce_loss_decoder.item(), global_step)
        # writer.add_scalar("loss/L1_Loss_Decoder1", l1_loss_decoder1.item(), global_step)
        # writer.add_scalar("loss/L2_Loss_Decoder1", l2_loss_decoder1.item(), global_step)
        # writer.add_scalar("loss/LPIPS_Loss_Decoder1", lpips_loss_decoder1.item(), global_step)
        # writer.add_scalar("loss/BCE_Loss_Decoder2", bce_loss_decoder2.item(), global_step)
        # writer.add_scalar("loss/Discriminator_Loss", D_loss.item(), global_step)
        writer.add_scalar("loss/Learning_Rate", scheduler.get_last_lr()[0], global_step)
        # writer.add_scalar("loss/Random_Transform", rnd_tran, global_step)

        global_step += 1
        if global_step == 1:
            print("Started training...")
        if global_step % 10 == 0:
            print(f"Step {global_step}, Total Loss: {total_loss.item():.4f}")
        if global_step % 200 == 0:
            torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
            torch.save(encoder1, os.path.join(args.saved_models, "encoder1.pth"))
            torch.save(decoder, os.path.join(args.saved_models, "decoder.pth"))
            torch.save(decoder1, os.path.join(args.saved_models, "decoder1.pth"))
            torch.save(decoder2, os.path.join(args.saved_models, "decoder2.pth"))

        scheduler.step()

    writer.close()
    # torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
    # torch.save(encoder1, os.path.join(args.saved_models, "encoder1.pth"))
    # torch.save(decoder, os.path.join(args.saved_models, "decoder.pth"))
    # torch.save(decoder1, os.path.join(args.saved_models, "decoder1.pth"))
    # torch.save(decoder2, os.path.join(args.saved_models, "decoder2.pth"))


if __name__ == '__main__':
    main()

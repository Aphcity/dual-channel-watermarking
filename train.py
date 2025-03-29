import os
import yaml
import random
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

    dataset = MyData(args.train_path, args.secret_size, size=(400, 400))
    
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

    height = 400
    width = 400

    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0


    

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            image_input, secret_input = next(iter(dataloader))
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
        encoder_output = encoder((secret_input, image_input))  
        encoded_image = encoder_output
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
        final_encoded_image = encoded_image + encoder1_output  

        extracted_secret = decoder(final_encoded_image)  

        reconstructed_residual = decoder1(final_encoded_image) 
        reconstructed_image = final_encoded_image - reconstructed_residual  

        extracted_hash = decoder2(reconstructed_image)  

        rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
        rnd_tran = np.random.uniform() * rnd_tran 
        Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
        if args.cuda:
            Ms = Ms.cuda()
        

        lpips_loss_encoder = lpips_alex(image_input, encoded_image).mean()
        lpips_loss_encoder1 = lpips_alex(image_input, final_encoded_image).mean()
        l2_loss_encoder = torch.nn.functional.mse_loss(image_input, encoded_image)
        l2_loss_encoder1 = torch.nn.functional.mse_loss(image_input, final_encoded_image)
        
   
        bce_loss_decoder = torch.nn.functional.binary_cross_entropy_with_logits(extracted_secret, secret_input)

        l1_loss_decoder1 = torch.nn.functional.l1_loss(reconstructed_residual, encoder1_output)
        l2_loss_decoder1 = torch.nn.functional.mse_loss(reconstructed_residual, encoder1_output)
        lpips_loss_decoder1 = lpips_alex(reconstructed_residual, encoder1_output).mean()
        
        D_loss = discriminator_loss(discriminator, final_encoded_image, image_input)
        print(secret_input.shape)
        bce_loss_decoder2 = torch.nn.functional.binary_cross_entropy_with_logits(extracted_hash, sha3_hash_tensor_batch)

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

        global_step += 1
        if global_step == 1:
            print("Started training...")
        if global_step % 10 == 0:
            print(f"Step {global_step}, Total Loss: {total_loss.item():.4f}")

    writer.close()
    torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
    torch.save(encoder1, os.path.join(args.saved_models, "encoder1.pth"))
    torch.save(decoder, os.path.join(args.saved_models, "decoder.pth"))
    torch.save(decoder1, os.path.join(args.saved_models, "decoder1.pth"))
    torch.save(decoder2, os.path.join(args.saved_models, "decoder2.pth"))


if __name__ == '__main__':
    main()

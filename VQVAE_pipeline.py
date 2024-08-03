import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_image.vq_loss import VQLoss


import argparse
import os
import time


def main(args): 
    # Hyperparameters
    device = torch.device("cuda")

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    )

    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,  
    ).to(device)

    print("hello world")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #VQVAE Model
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    
    #VQ_Loss
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    
    
    #Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)


    args = parser.parse_args()
    main(args)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from accelerate import Accelerator

from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_image.vq_loss import VQLoss

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset

from PIL import Image
import argparse


def main(args): 
    # Hyperparameters
    device = torch.device("cuda")

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    ).to(device)

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

    optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, 512)), #얘네가 직접 만든 random_crop_arr 함수
        transforms.RandomHorizontalFlip(), #transforms 라이브러리에 있는 함수
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    print("now loading dataset")
    dataset = build_dataset(args, transform=transform)
    print("dataset loaded")


    train_loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle = True,
        #num_workers=args.num_workers, #여기에서 에러
        pin_memory=True,
        drop_last=True
    )


    vq_model, vq_loss, optimizer, optimizer_disc, train_loader = accelerator.prepare(
        vq_model, vq_loss, optimizer, optimizer_disc, train_loader
    )

    print("FINISHED SETUP")

    for data, _ in train_loader:
        img = data[0]
        print(img.shape)
        break;

    quit()

    ########################################################################
    # Training
    ########################################################################

    for epoch in range(args.epochs):
        for batch in train_loader:
            imgs, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            #generator training
            optimizer.zero_grad()
            with accelerator.autocast():
                recons_imgs, codebook_loss = vq_model(imgs)
                loss_gen = vq_loss(
                    codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=0, 
                    last_layer=vq_model.module.decoder.last_layer, 
                    logger=None, log_every=args.log_every
                )
            accelerator.backward(loss_gen)
            optimizer.step()

            #discriminator training
            optimizer_disc.zero_grad()
            with accelerator.autocast():
                loss_disc = vq_loss(
                    codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=0,
                    logger=None, log_every=args.log_every
                )
            accelerator.backward(loss_disc)
            optimizer_disc.step()
            
            print("loop finished")

    print("End of training")


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
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    #Dataset
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--data-path", type=str, default='E:\imagenet1k_train') #required=True
   

    #etc
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 


    args = parser.parse_args()
    main(args)

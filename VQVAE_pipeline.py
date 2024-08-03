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
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    num_workers = 4
    save_interval = 10


    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    
    args = parser.parse_args()
    main(args)

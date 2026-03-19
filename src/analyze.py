from operator import truediv
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator


# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models import *

# accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = "cuda:0"
print("device:",device)

parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default="/data20TB/lzg/MindEyeV2",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default="/data20TB/lzg/MindEyeV2",
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=1024,
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)

args = parser.parse_args()

    
# seed all random functions
utils.seed_everything(args.seed)

# make output directory
os.makedirs("evals",exist_ok=True)
os.makedirs(f"evals/{args.model_name}",exist_ok=True)

voxels = {}
# Load hdf5 data for betas
f = h5py.File(f'{args.data_path}/betas_all_subj0{args.subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{args.subj}'] = betas
print(f"num_voxels for subj0{args.subj}: {num_voxels}")

subj = args.subj
if not args.new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{args.data_path}/wds/subj0{subj}/test/" + "0.tar"
else: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{args.data_path}/wds/subj0{subj}/new_test/" + "0.tar"
    
print(test_url)
def my_split_by_node(urls): return urls
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

# Prep images but don't load them all to memory
f = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
    test_voxels_idx = np.append(test_images_idx, behav[:,0,5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664

if args.blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{args.cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
        return out
        
model.ridge = RidgeRegression([num_voxels], out_features=args.hidden_dim)

from diffusers.models.autoencoders.vae import Decoder
from model_ori import *
model.backbone = BrainNetwork(
        h=args.hidden_dim,
        in_dim=args.hidden_dim,
        seq_len=1,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=args.blurry_recon,
    )
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device)

utils.count_params(model.diffusion_prior)
utils.count_params(model)

tag='last'
outdir = os.path.abspath('/data20TB/lzg/MindEyeV2/train_logs/final_subj02_pretrained_40sess_24bs')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")

checkpoint = torch.load('/data20TB/lzg/MindEyeV2/train_logs/final_subj02_pretrained_40sess_24bs/last.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict, strict=True)
del checkpoint
print("ckpt loaded!")



batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device)}

# get all reconstructions
model.to(device)
model.eval().requires_grad_(False)

# all_images = None
all_blurryrecons = None
all_recons = None
all_predcaptions = []
all_clipvoxels = None

minibatch_size = 1
num_samples_per_image = 1
assert num_samples_per_image == 1

#if utils.is_interactive(): plotting=True
save_dir = "evals/diff_analysis_final"
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
    for batch in tqdm(range(0,len(np.unique(test_images_idx)),minibatch_size)):
        uniq_imgs = np.unique(test_images_idx)[batch:batch+minibatch_size]
        voxel = None
        for uniq_img in uniq_imgs:
            locs = np.where(test_images_idx==uniq_img)[0]
            if len(locs)==1:
                locs = locs.repeat(3)
            elif len(locs)==2:
                locs = locs.repeat(2)[:3]
            assert len(locs)==3
            if voxel is None:
                voxel = test_voxels[None,locs] # 1, num_image_repetitions, num_voxels
            else:
                voxel = torch.vstack((voxel, test_voxels[None,locs]))
        voxel = voxel.to(device)
        
        batch_imgs = []
        for img_idx in uniq_imgs:
            img = images[img_idx]  # 从 HDF5 里取出 numpy 图像 [3,H,W]
            img = torch.tensor(img).float() / 255.0  # 转为 float tensor [0,1]
            batch_imgs.append(img)
        batch_imgs = torch.stack(batch_imgs).to(device) 

        clip_emb = clip_img_embedder(batch_imgs) 
        
        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:,[rep]],0) # 0th index of subj_list
            backbone0, clip_voxels0, _ = model.backbone(voxel_ridge)
            if rep==0:
                clip_voxels = clip_voxels0
                backbone = backbone0
            else:
                clip_voxels += clip_voxels0
                backbone += backbone0
        clip_voxels /= 3
        backbone /= 3
        
        # Feed voxels through OpenCLIP-bigG diffusion prior
        prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                        text_cond = dict(text_embed = backbone), 
                        cond_scale = 1., timesteps = 20)
        
        diff = clip_emb - prior_out
        
        clip_fft = np.abs(np.fft.fftshift(np.fft.fft2(clip_emb.cpu().numpy())))
        prior_fft = np.abs(np.fft.fftshift(np.fft.fft2(prior_out.cpu().numpy())))
        diff_fft = np.abs(np.fft.fftshift(np.fft.fft2(diff.cpu().numpy())))
        
        for i, img_idx in enumerate(uniq_imgs):
            fig, axs = plt.subplots(1,4, figsize=(36,8))

            im0 = axs[0].imshow(clip_fft[i], aspect='auto', cmap='inferno')
            axs[0].set_title("CLIP Embedding fft")
            fig.colorbar(im0, ax=axs[0], fraction=0.1, pad=0.02)
            im1 = axs[1].imshow(prior_fft[i], aspect='auto', cmap='inferno')
            axs[1].set_title("Diffusion Prior Output fft")
            fig.colorbar(im1, ax=axs[1], fraction=0.1, pad=0.02)
            im2 = axs[2].imshow(diff[i].cpu(), aspect='auto', cmap='seismic', vmin=-diff.abs().max(), vmax=diff.abs().max())
            axs[2].set_title("Difference (CLIP - Prior)")
            fig.colorbar(im2, ax=axs[2], fraction=0.1, pad=0.02)
            im3 = axs[3].imshow(diff_fft[i], aspect='auto', cmap='inferno')
            axs[3].set_title("FFT of Difference")
            fig.colorbar(im3, ax=axs[3], fraction=0.1, pad=0.02)

            plt.tight_layout()
            plt.savefig(f"evals/diff_analysis_final/img_{img_idx}.png", dpi=400)
            plt.close(fig)
import os
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
import wandb
import kornia
from kornia.augmentation.container import AugmentationSequential
from transformers import AutoModel, AutoImageProcessor

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from args import get_train_args
from models import *

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append("generative_models/")
import sgm
from generative_models.sgm.modules.encoders.modules import (
    FrozenOpenCLIPImageEmbedder,
)  # bigG embedder

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils

device0 = "cuda:0"
device1 = "cuda:1"
device2 = "cuda:2"
device3 = "cuda:3"
device = device0

args = get_train_args()
outdir = os.path.abspath(f"../train_logs/{args.model_name}")


def save_ckpt(tag, model, optimizer, epoch, lr_scheduler, losses, test_losses, lrs):
    ckpt_path = outdir + f"/{tag}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "train_losses": losses,
            "test_losses": test_losses,
            "lrs": lrs,
        },
        ckpt_path,
    )
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")


def load_ckpt(
    tag,
    model,
    epoch,
    optimizer,
    lr_scheduler,
    load_lr=True,
    load_optimizer=True,
    load_epoch=True,
    strict=True,
    outdir=outdir,
    multisubj_loading=False,
):
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir + "/last.pth", map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    if multisubj_loading:  # remove incompatible ridge layer that will otherwise error
        state_dict.pop("ridge.linears.0.weight", None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint["epoch"]
        print("Epoch", epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    del checkpoint


def main():
    utils.seed_everything(args.seed)
    data_type = torch.float32

    # Configurations
    if os.path.exists(outdir) is False:
        os.makedirs(outdir, exist_ok=True)

    if args.use_image_aug:
        img_augment = AugmentationSequential(
            kornia.augmentation.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3
            ),
            same_on_batch=False,
            data_keys=["input"],
        )

    if args.multi_subject:
        subj_list = np.arange(1, 9)
        subj_list = subj_list[subj_list != args.subj]
    else:
        subj_list = [args.subj]

    print("subj_list", subj_list, "num_sessions", args.num_sessions)

    if args.multi_subject:
        nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
        num_samples_per_epoch = 750 * 40
    else:
        num_samples_per_epoch = 750 * args.num_sessions

    print(
        "dividing batch size by subj_list, which will then be concatenated across subj during training..."
    )
    batch_size = args.batch_size // len(subj_list)

    num_iterations_per_epoch = num_samples_per_epoch // (batch_size * len(subj_list))

    print(
        "batch_size =",
        batch_size,
        "num_iterations_per_epoch =",
        num_iterations_per_epoch,
        "num_samples_per_epoch =",
        num_samples_per_epoch,
    )

    # Prepare data, models and dataloaders
    train_data = {}
    train_dl = {}
    num_voxels = {}
    voxels = {}
    num_voxels_list = []

    for s in subj_list:
        print(f"Training with {args.num_sessions} sessions for subject {s}")
        if args.multi_subject:
            train_url = f"{args.data_path}/wds/subj0{s}/train/{{0..{nsessions_allsubj[s-1]-1}}}.tar"
        else:
            train_url = f"{args.data_path}/wds/subj0{s}/train/{{0..{args.num_sessions - 1}}}.tar"
        print(f"train_url: {train_url}")

        train_data[f"subj0{s}"] = (
            wds.WebDataset(train_url, resampled=True)
            .shuffle(750, initial=1500, rng=random.Random(42))
            .decode("torch")
            .rename(
                behav="behav.npy",
                past_behav="past_behav.npy",
                future_behav="future_behav.npy",
                olds_behav="olds_behav.npy",
            )
            .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
        )
        train_dl[f"subj0{s}"] = torch.utils.data.DataLoader(
            train_data[f"subj0{s}"],
            batch_size=args.batch_size,
            drop_last=False,
            pin_memory=True,
        )

        f = h5py.File(f"{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5", "r")
        betas = f["betas"][:]
        betas = torch.Tensor(betas).to("cpu").to(data_type)
        num_voxels_list.append(betas[0].shape[-1])
        num_voxels[f"subj0{s}"] = betas[0].shape[-1]
        voxels[f"subj0{s}"] = betas
        print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

    print("Loaded all subj train dls and betas!\n")

    # Validate only on one subject
    subj = args.subj
    if args.multi_subject:
        subj = subj_list[0]
    if not args.new_test:
        if subj == 3:
            num_test = 2113
        elif subj == 4:
            num_test = 1985
        elif subj == 6:
            num_test = 2113
        elif subj == 8:
            num_test = 1985
        else:
            num_test = 2770
        test_url = f"{args.data_path}/wds/subj0{subj}/test/" + "0.tar"
    elif args.new_test:  # using larger test set from after full dataset released
        if subj == 3:
            num_test = 2371
        elif subj == 4:
            num_test = 2188
        elif subj == 6:
            num_test = 2371
        elif subj == 8:
            num_test = 2188
        else:
            num_test = 3000
        test_url = f"{args.data_path}/wds/subj0{subj}/new_test/" + "0.tar"
    print(f"test_url: {test_url}")

    test_data = (
        wds.WebDataset(test_url, resampled=False)
        .shuffle(750, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    )
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True
    )
    print(f"Loaded test dl for subj{subj}!\n")

    # Load 73k NSD images
    f = h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r")
    images = f["images"]
    print("Loaded all 73k possible NSD images to cpu!", images.shape)

    # Load models
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)

    sam_img_embedder = AutoencoderKL.from_pretrained("facebook/sam2")
    sam_img_processor = AutoImageProcessor.from_pretrained("facebook/sam2")

    clip_seq_dim = 256
    clip_emb_dim = 1664

    if args.blurry_recon:
        from diffusers import AutoencoderKL

        autoenc = AutoencoderKL(
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            sample_size=256,
        )
        ckpt = torch.load(f"{args.cache_dir}/sd_image_var_autoenc.pth")
        autoenc.load_state_dict(ckpt)

        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(device)
        # utils.count_params(autoenc)

        from autoencoder.convnext import ConvnextXL

        cnx = ConvnextXL(f"{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth")
        cnx.requires_grad_(False)
        cnx.eval()
        cnx.to(device)

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1, 3, 1, 1)

        blur_augs = AugmentationSequential(
            kornia.augmentation.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            kornia.augmentation.RandomGrayscale(p=0.1),
            kornia.augmentation.RandomSolarize(p=0.1),
            kornia.augmentation.RandomResizedCrop(
                (224, 224), scale=(0.9, 0.9), ratio=(1, 1), p=1.0
            ),
            data_keys=["input"],
        )

    class MindEyeModule(nn.Module):
        def __init__(self):
            super(MindEyeModule, self).__init__()

        def forward(self, x):
            return x

    model = MindEyeModule()

    class RidgeRegression(nn.Module):
        def __init__(self, input_sizes, out_features):
            super(RidgeRegression, self).__init__()
            self.out_features = out_features
            self.linears = torch.nn.ModuleList(
                [
                    torch.nn.Linear(input_size, out_features)
                    for input_size in input_sizes
                ]
            )

        def forward(self, x, subj_idx):
            out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
            return out

    model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim)
    # utils.count_params(model.ridge)
    # utils.count_params(model)

    from models import BrainNetwork

    model.backbone = BrainNetworkSpike(
        h=args.hidden_dim,
        in_dim=args.hidden_dim,
        seq_len=1,
        n_blocks=args.n_blocks,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=args.blurry_recon,
        clip_scale=args.clip_scale,
        T=args.T,
    )

    """model.backbone = BrainNetwork(
        h=args.hidden_dim,
        in_dim=args.hidden_dim,
        seq_len=1,
        n_blocks=args.n_blocks,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=args.blurry_recon,
        clip_scale=args.clip_scale,
    )"""
    # utils.count_params(model.backbone)
    # utils.count_params(model)

    if args.use_prior:
        # setup diffusion prior network
        out_dim = clip_emb_dim
        depth = 6
        dim_head = 52
        heads = clip_emb_dim // 52
        timesteps = 100

        prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens=clip_seq_dim,
            learned_query_mode="pos_emb",
        )
        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )
        # utils.count_params(model.diffusion_prior)
        # utils.count_params(model)
    model = model.to(device)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    opt_grouped_parameters = [
        {
            "params": [p for n, p in model.ridge.named_parameters()],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p
                for n, p in model.backbone.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p
                for n, p in model.backbone.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if args.use_prior:
        opt_grouped_parameters.extend(
            [
                {
                    "params": [
                        p
                        for n, p in model.diffusion_prior.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 1e-2,
                },
                {
                    "params": [
                        p
                        for n, p in model.diffusion_prior.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        )

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)

    if args.lr_scheduler_type == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.num_epochs * num_iterations_per_epoch)),
            last_epoch=-1,
        )
    elif args.lr_scheduler_type == "cycle":
        total_steps = int(np.floor(args.num_epochs * num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1,
            pct_start=2 / args.num_epochs,
        )

    print("\nDone with model preparations!")
    num_params = utils.count_params(model)
    print(torch.cuda.memory_summary(device=1))

    if args.wandb_log:  # only use main process for wandb logging
        wandb_project = "mindeye"
        print(f"wandb {wandb_project} run {args.model_name}")
        # need to configure wandb beforehand in terminal with "wandb init"!
        wandb_config = {
            "model_name": args.model_name,
            # "global_batch_size": global_batch_size,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs,
            "num_sessions": args.num_sessions,
            "num_params": num_params,
            "clip_scale": args.clip_scale,
            "prior_scale": args.prior_scale,
            "blur_scale": args.blur_scale,
            "use_image_aug": args.use_image_aug,
            "max_lr": args.max_lr,
            "mixup_pct": args.mixup_pct,
            "num_samples_per_epoch": num_samples_per_epoch,
            "num_test": num_test,
            "ckpt_interval": args.ckpt_interval,
            "ckpt_saving": args.ckpt_saving,
            "seed": args.seed,
            # "distributed": distributed,
            # "num_devices": num_devices,
            # "world_size": world_size,
            "train_url": train_url,
            "test_url": test_url,
        }
        print("wandb_config:\n", wandb_config)
        print("wandb_id:", args.model_name)
        wandb.init(
            id=args.model_name,
            project=wandb_project,
            name=args.model_name,
            config=wandb_config,
            resume="allow",
        )
    else:
        wandb_log = False

    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test_loss = 1e9
    torch.cuda.empty_cache()

    # load multisubject stage1 ckpt if set
    if args.multisubject_ckpt is not None:
        load_ckpt(
            "last",
            outdir=args.multisubject_ckpt,
            load_lr=False,
            load_optimizer=False,
            load_epoch=False,
            strict=False,
            multisubj_loading=True,
        )

    train_dls = [train_dl[f"subj0{s}"] for s in subj_list]

    print(f"{args.model_name} starting with epoch {epoch} / {args.num_epochs}")
    progress_bar = tqdm(range(epoch, args.num_epochs), ncols=1200)
    test_image, test_voxel = None, None
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(
        0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs)
    )

    for epoch in progress_bar:
        model.train()

        fwd_percent_correct = 0.0
        bwd_percent_correct = 0.0
        test_fwd_percent_correct = 0.0
        test_bwd_percent_correct = 0.0

        recon_cossim = 0.0
        test_recon_cossim = 0.0
        recon_mse = 0.0
        test_recon_mse = 0.0

        loss_clip_total = 0.0
        loss_blurry_total = 0.0
        loss_blurry_cont_total = 0.0
        test_loss_clip_total = 0.0

        loss_prior_total = 0.0
        test_loss_prior_total = 0.0

        blurry_pixcorr = 0.0
        test_blurry_pixcorr = (
            0.0  # needs >.456 to beat low-level subj01 results in mindeye v1
        )

        # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
        voxel_iters = {}  # empty dict because diff subjects have differing # of voxels
        image_iters = torch.zeros(
            num_iterations_per_epoch, batch_size * len(subj_list), 3, 224, 224
        ).float()
        annot_iters = {}
        perm_iters, betas_iters, select_iters = {}, {}, {}
        for s, train_dl in enumerate(train_dls):
            with torch.amp.autocast("cuda", dtype=data_type):
                iter = -1
                for behav0, past_behav0, future_behav0, old_behav0 in train_dl:
                    # Load images to cpu from hdf5 (requires sorted indexing)
                    image_idx = behav0[:, 0, 0].cpu().long().numpy()
                    image0, image_sorted_idx = np.unique(image_idx, return_index=True)
                    if len(image0) != len(
                        image_idx
                    ):  # hdf5 cant handle duplicate indexing
                        continue
                    iter += 1
                    image0 = torch.tensor(images[image0], dtype=data_type)
                    image_iters[iter, s * batch_size : s * batch_size + batch_size] = (
                        image0
                    )

                    # Load voxels for current batch, matching above indexing
                    voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
                    voxel_sorted_idx = voxel_idx[image_sorted_idx]
                    voxel0 = voxels[f"subj0{subj_list[s]}"][voxel_sorted_idx]
                    voxel0 = torch.Tensor(voxel0).unsqueeze(1)

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        voxel0, perm, betas, select = utils.mixco(voxel0)
                        perm_iters[f"subj0{subj_list[s]}_iter{iter}"] = perm
                        betas_iters[f"subj0{subj_list[s]}_iter{iter}"] = betas
                        select_iters[f"subj0{subj_list[s]}_iter{iter}"] = select

                    voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

                    if iter >= num_iterations_per_epoch - 1:
                        break

        # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
        for train_i in range(num_iterations_per_epoch):
            with torch.amp.autocast("cuda", dtype=data_type):
                optimizer.zero_grad()
                loss = 0.0

                voxel_list = [
                    voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device)
                    for s in subj_list
                ]
                image = image_iters[train_i].detach()
                image = image.to(device)

                if args.use_image_aug:
                    image = img_augment(image)

                clip_target = clip_img_embedder(image)
                sam_target = sam_img_embedder(image)
                assert not torch.any(torch.isnan(clip_target))

                if epoch < int(args.mixup_pct * args.num_epochs):
                    perm_list = [
                        perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device)
                        for s in subj_list
                    ]
                    perm = torch.cat(perm_list, dim=0)
                    betas_list = [
                        betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device)
                        for s in subj_list
                    ]
                    betas = torch.cat(betas_list, dim=0)
                    select_list = [
                        select_iters[f"subj0{s}_iter{train_i}"].detach().to(device)
                        for s in subj_list
                    ]
                    select = torch.cat(select_list, dim=0)

                voxel_ridge_list = [
                    model.ridge(voxel_list[si], si) for si, s in enumerate(subj_list)
                ]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
                # print(f"shape:{voxel_ridge.shape}")

                backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)
                # print(f"blurry {blurry_image_enc_.shape}")

                if args.clip_scale > 0:
                    clip_voxels_norm = nn.functional.normalize(
                        clip_voxels.flatten(1), dim=-1
                    )
                    clip_target_norm = nn.functional.normalize(
                        clip_target.flatten(1), dim=-1
                    )

                if args.use_prior:
                    # print(f"backbone\n {backbone.shape}\n clip_target\n {clip_target.shape}")
                    loss_prior, prior_out = model.diffusion_prior(
                        text_embed=backbone, image_embed=clip_target
                    )
                    loss_prior_total += loss_prior.item()
                    loss_prior *= args.prior_scale
                    loss += loss_prior

                    recon_cossim += (
                        nn.functional.cosine_similarity(prior_out, clip_target)
                        .mean()
                        .item()
                    )
                    recon_mse += mse(prior_out, clip_target).item()

                if args.clip_scale > 0:
                    if epoch < int(args.mixup_pct * args.num_epochs):
                        loss_clip = utils.mixco_nce(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=0.006,
                            perm=perm,
                            betas=betas,
                            select=select,
                        )
                    else:
                        epoch_temp = soft_loss_temps[
                            epoch - int(args.mixup_pct * args.num_epochs)
                        ]
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm, clip_target_norm, temp=epoch_temp
                        )

                    loss_clip_total += loss_clip.item()
                    loss_clip *= args.clip_scale
                    loss += loss_clip

                if args.blurry_recon:
                    image_enc_pred, transformer_feats = blurry_image_enc_

                    image_enc = (
                        autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
                    )
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_total += loss_blurry.item()

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        image_enc_shuf = image_enc[perm]
                        betas_shape = [-1] + [1] * (len(image_enc.shape) - 1)
                        image_enc[select] = image_enc[select] * betas[select].reshape(
                            *betas_shape
                        ) + image_enc_shuf[select] * (1 - betas[select]).reshape(
                            *betas_shape
                        )

                    image_norm = (image - mean) / std
                    image_aug = (blur_augs(image) - mean) / std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)

                    cont_loss = utils.soft_cont_loss(
                        nn.functional.normalize(
                            transformer_feats.reshape(-1, transformer_feats.shape[-1]),
                            dim=-1,
                        ),
                        nn.functional.normalize(
                            cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1
                        ),
                        nn.functional.normalize(
                            cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1
                        ),
                        temp=0.2,
                    )
                    loss_blurry_cont_total += cont_loss.item()

                    loss += (loss_blurry + 0.1 * cont_loss) * args.blur_scale  # /.18215

                if args.clip_scale > 0:
                    # forward and backward top 1 accuracy
                    labels = torch.arange(len(clip_voxels_norm)).to(
                        clip_voxels_norm.device
                    )
                    fwd_percent_correct += utils.topk(
                        utils.batchwise_cosine_similarity(
                            clip_voxels_norm, clip_target_norm
                        ),
                        labels,
                        k=1,
                    ).item()
                    bwd_percent_correct += utils.topk(
                        utils.batchwise_cosine_similarity(
                            clip_target_norm, clip_voxels_norm
                        ),
                        labels,
                        k=1,
                    ).item()

                if args.blurry_recon:
                    with torch.no_grad():
                        # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                        random_samps = np.random.choice(
                            np.arange(len(image)), size=len(image) // 5, replace=False
                        )
                        blurry_recon_images = (
                            autoenc.decode(
                                image_enc_pred[random_samps] / 0.18215
                            ).sample
                            / 2
                            + 0.5
                        ).clamp(0, 1)
                        pixcorr = utils.pixcorr(
                            image[random_samps], blurry_recon_images
                        )
                        blurry_pixcorr += pixcorr.item()

                utils.check_loss(loss)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])

                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=data_type):
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(
                test_dl
            ):
                # all test samples should be loaded per batch such that test_i should never exceed 0
                assert len(behav) == num_test

                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels[f"subj0{subj}"][
                        behav[:, 0, 5].cpu().long()
                    ].unsqueeze(1)

                    image = behav[:, 0, 0].cpu().long()

                    unique_image, sort_indices = torch.unique(
                        image, return_inverse=True
                    )
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if len(locs) == 1:
                            locs = locs.repeat(3)
                        elif len(locs) == 2:
                            locs = locs.repeat(2)[:3]
                        assert len(locs) == 3
                        if test_image is None:
                            test_image = torch.Tensor(images[im][None])
                            test_voxel = voxel[locs][None]
                        else:
                            test_image = torch.vstack(
                                (test_image, torch.Tensor(images[im][None]))
                            )
                            test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                loss = 0.0

                test_indices = torch.arange(len(test_voxel))[:300]
                voxel = test_voxel[test_indices].to(device)
                image = test_image[test_indices].to(device)
                assert len(image) == 300

                clip_target = clip_img_embedder(image.float())

                for rep in range(3):
                    voxel_ridge = model.ridge(
                        voxel[:, rep], 0
                    )  # 0th index of subj_list
                    backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(
                        voxel_ridge
                    )
                    if rep == 0:
                        clip_voxels = clip_voxels0
                        backbone = backbone0
                    else:
                        clip_voxels += clip_voxels0
                        backbone += backbone0
                clip_voxels /= 3
                backbone /= 3

                if args.clip_scale > 0:
                    clip_voxels_norm = nn.functional.normalize(
                        clip_voxels.flatten(1), dim=-1
                    )
                    clip_target_norm = nn.functional.normalize(
                        clip_target.flatten(1), dim=-1
                    )

                # for some evals, only doing a subset of the samples per batch because of computational cost
                random_samps = np.random.choice(
                    np.arange(len(image)), size=len(image) // 5, replace=False
                )

                if args.use_prior:
                    loss_prior, contaminated_prior_out = model.diffusion_prior(
                        text_embed=backbone[random_samps],
                        image_embed=clip_target[random_samps],
                    )
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= args.prior_scale
                    loss += loss_prior

                if args.clip_scale > 0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm, clip_target_norm, temp=0.006
                    )

                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * args.clip_scale
                    loss += loss_clip

                if args.blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (
                        autoenc.decode(image_enc_pred[random_samps] / 0.18215).sample
                        / 2
                        + 0.5
                    ).clamp(0, 1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    test_blurry_pixcorr += pixcorr.item()

                if args.clip_scale > 0:
                    # forward and backward top 1 accuracy
                    labels = torch.arange(len(clip_voxels_norm)).to(
                        clip_voxels_norm.device
                    )
                    test_fwd_percent_correct += utils.topk(
                        utils.batchwise_cosine_similarity(
                            clip_voxels_norm, clip_target_norm
                        ),
                        labels,
                        k=1,
                    ).item()
                    test_bwd_percent_correct += utils.topk(
                        utils.batchwise_cosine_similarity(
                            clip_target_norm, clip_voxels_norm
                        ),
                        labels,
                        k=1,
                    ).item()

                utils.check_loss(loss)
                test_losses.append(loss.item())

            assert (test_i + 1) == 1
            logs = {
                "train/loss": np.mean(losses[-(train_i + 1) :]),
                "test/loss": np.mean(test_losses[-(test_i + 1) :]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                "train/recon_cossim": recon_cossim / (train_i + 1),
                "test/recon_cossim": test_recon_cossim / (test_i + 1),
                "train/recon_mse": recon_mse / (train_i + 1),
                "test/recon_mse": test_recon_mse / (test_i + 1),
                "train/loss_prior": loss_prior_total / (train_i + 1),
                "test/loss_prior": test_loss_prior_total / (test_i + 1),
            }

            # if finished training, save jpg recons if they exist
            if (epoch == args.num_epochs - 1) or (epoch % args.ckpt_interval == 0):
                if args.blurry_recon:
                    image_enc = (
                        autoenc.encode(2 * image[:4] - 1).latent_dist.mode() * 0.18215
                    )
                    # transform blurry recon latents to images and plot it
                    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                    jj = -1
                    for j in [0, 1, 2, 3]:
                        jj += 1
                        axes[jj].imshow(
                            utils.torch_to_Image(
                                (
                                    autoenc.decode(image_enc[[j]] / 0.18215).sample / 2
                                    + 0.5
                                ).clamp(0, 1)
                            )
                        )
                        axes[jj].axis("off")
                        jj += 1
                        axes[jj].imshow(
                            utils.torch_to_Image(
                                (
                                    autoenc.decode(image_enc_pred[[j]] / 0.18215).sample
                                    / 2
                                    + 0.5
                                ).clamp(0, 1)
                            )
                        )
                        axes[jj].axis("off")

                    if wandb_log:
                        logs[f"test/blur_recons"] = wandb.Image(
                            fig, caption=f"epoch{epoch:03d}"
                        )
                        plt.close()
                    else:
                        plt.show()

            progress_bar.set_postfix(**logs)

            if wandb_log:
                wandb.log(logs)

        # Save model checkpoint and reconstruct
        if (args.ckpt_saving) and (epoch % args.ckpt_interval == 0):
            save_ckpt(
                f"last", model, optimizer, epoch, lr_scheduler, losses, test_losses, lrs
            )

        torch.cuda.empty_cache()

    print("\n===Finished!===\n")
    if args.ckpt_saving:
        save_ckpt(
            f"last", model, optimizer, epoch, lr_scheduler, losses, test_losses, lrs
        )

    # In[ ]:
    plt.plot(losses)
    plt.show()
    plt.plot(test_losses)
    plt.show()


if __name__ == "__main__":
    main()

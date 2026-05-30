import argparse
import os
import sys
import random
import h5py
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
from tqdm import tqdm
from diffusers import AutoencoderKL
from kornia.augmentation.container import AugmentationSequential

import utils
from models import BrainDiffusionPrior, BrainNetwork, PriorNetwork

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append("generative_models/")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

torch.backends.cuda.matmul.allow_tf32 = True


def my_split_by_node(urls):
    return urls


def build_train_parser():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument("--model_name", type=str, default="testing", help="name of model, used for ckpt saving and wandb logging")
    parser.add_argument("--data_path", type=str, default=os.getcwd(), help="path to where NSD data is stored / where to download it to")
    parser.add_argument("--cache_dir", type=str, default=os.getcwd(), help="path to misc files downloaded from Hugging Face")
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8], help="validate on which subject")
    parser.add_argument("--multisubject_ckpt", type=str, default=None, help="pre-trained multisubject model to finetune from")
    parser.add_argument("--num_sessions", type=int, default=1, help="number of training sessions to include")
    parser.add_argument("--use_prior", action=argparse.BooleanOptionalAction, default=True, help="train diffusion prior")
    parser.add_argument("--batch_size", type=int, default=16, help="per-step global batch size before subject split")
    parser.add_argument("--wandb_log", action=argparse.BooleanOptionalAction, default=False, help="whether to log to wandb")
    parser.add_argument("--wandb_project", type=str, default="mindeye", help="wandb project name")
    parser.add_argument("--mixup_pct", type=float, default=0.33, help="when to switch from BiMixCo to SoftCLIP")
    parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True, help="whether to output blurry reconstructions")
    parser.add_argument("--blur_scale", type=float, default=0.5, help="multiply blurry reconstruction loss")
    parser.add_argument("--clip_scale", type=float, default=1.0, help="multiply contrastive loss")
    parser.add_argument("--prior_scale", type=float, default=30, help="multiply diffusion prior loss")
    parser.add_argument("--use_image_aug", action=argparse.BooleanOptionalAction, default=False, help="whether to use image augmentation")
    parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs of training")
    parser.add_argument("--multi_subject", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--lr_scheduler_type", type=str, default="cycle", choices=["cycle", "linear"])
    parser.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt_interval", type=int, default=5, help="save backup ckpt and reconstruct every x epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda", help="torch device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--outdir", type=str, default=None, help="checkpoint directory; defaults to ../train_logs/{model_name}")
    return parser

def save_ckpt(tag, model, optimizer, epoch, lr_scheduler, losses, test_losses, lrs, outdir):
    ckpt_path = os.path.join(outdir, f"{tag}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "train_losses": losses,
            "test_losses": test_losses,
            "lrs": lrs,
        },
        ckpt_path,
    )
    print(f"saved checkpoint: {ckpt_path}")


def load_ckpt(tag, model, optimizer=None, lr_scheduler=None, load_lr=True, load_optimizer=True, load_epoch=True, strict=True, outdir=None, multisubj_loading=False):
    ckpt_path = os.path.join(outdir, f"{tag}.pth")
    print(f"loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    if multisubj_loading:
        state_dict.pop("ridge.linears.0.weight", None)
    model.load_state_dict(state_dict, strict=strict)
    epoch = checkpoint.get("epoch", 0) if load_epoch else 0
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if load_lr and lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return epoch, checkpoint


def make_image_augment():
    return AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )


def make_blur_augment():
    return AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224, 224), scale=(0.9, 0.9), ratio=(1, 1), p=1.0),
        data_keys=["input"],
    )

def create_dataset(url, resampled):
    dataset = wds.WebDataset(url, resampled=resampled, nodesplitter=my_split_by_node)
    dataset = dataset.shuffle(750, initial=1500, rng=random.Random(42))
    return dataset.decode("torch").rename(
        behav="behav.npy",
        past_behav="past_behav.npy",
        future_behav="future_behav.npy",
        olds_behav="olds_behav.npy",
    ).to_tuple("behav", "past_behav", "future_behav", "olds_behav")


def load_train_dataloaders(args, subj_list, batch_size, data_type):
    nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
    train_dl, voxels, num_voxels_list, train_urls = {}, {}, [], {}
    for subj in subj_list:
        session_count = nsessions_allsubj[subj - 1] if args.multi_subject else args.num_sessions
        train_url = f"{args.data_path}/wds/subj0{subj}/train/{{0..{session_count - 1}}}.tar"
        print(f"subject {subj}: {train_url}")
        train_urls[f"subj0{subj}"] = train_url
        train_data = create_dataset(train_url, resampled=True)
        train_dl[f"subj0{subj}"] = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=False, pin_memory=True)
        with h5py.File(f"{args.data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
            betas = torch.tensor(f["betas"][:], dtype=data_type)
        voxels[f"subj0{subj}"] = betas
        num_voxels_list.append(int(betas.shape[-1]))
        print(f"subject {subj} num_voxels: {betas.shape[-1]}")
    return train_dl, voxels, num_voxels_list, train_urls

def load_test_dataloader(args, subj, split, num_test):
    test_url = f"{args.data_path}/wds/subj0{subj}/{split}/0.tar"
    print(f"test_url: {test_url}")
    test_data = create_dataset(test_url, resampled=False)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
    return test_dl, test_url, num_test


class MindEyeModule(nn.Module):
    def forward(self, x):
        return x


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, out_features) for input_size in input_sizes])

    def forward(self, x, subj_idx):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


def build_model(args, num_voxels_list, clip_emb_dim, clip_seq_dim):
    model = MindEyeModule()
    model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim)
    model.backbone = BrainNetwork(
        h=args.hidden_dim,
        in_dim=args.hidden_dim,
        seq_len=1,
        n_blocks=args.n_blocks,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=args.blurry_recon,
        clip_scale=args.clip_scale,
    )
    if args.use_prior:
        prior_network = PriorNetwork(
            dim=clip_emb_dim,
            depth=6,
            dim_head=52,
            heads=clip_emb_dim // 52,
            causal=False,
            num_tokens=clip_seq_dim,
            learned_query_mode="pos_emb",
        )
        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=clip_emb_dim,
            condition_on_text_encodings=False,
            timesteps=100,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )
    return model.to(args.device)


def build_recon_modules(args):
    if not args.blurry_recon:
        return {}
    autoenc = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    autoenc.load_state_dict(torch.load(f"{args.cache_dir}/sd_image_var_autoenc.pth", map_location="cpu"))
    autoenc.eval().requires_grad_(False).to(args.device)

    from autoencoder.convnext import ConvnextXL

    cnx = ConvnextXL(f"{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth")
    cnx.eval().requires_grad_(False).to(args.device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225], device=args.device).reshape(1, 3, 1, 1)
    return {"autoenc": autoenc, "cnx": cnx, "mean": mean, "std": std, "blur_augs": make_blur_augment()}


def build_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    grouped = [
        {"params": list(model.ridge.parameters()), "weight_decay": 1e-2},
        {"params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 1e-2},
        {"params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if hasattr(model, "diffusion_prior"):
        grouped.extend(
            [
                {"params": [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 1e-2},
                {"params": [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
        )
    return torch.optim.AdamW(grouped, lr=args.max_lr)


def build_scheduler(optimizer, args, steps_per_epoch):
    if args.lr_scheduler_type is None:
        return None
    total_steps = int(np.floor(args.num_epochs * steps_per_epoch))
    if args.lr_scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps, last_epoch=-1)
    pct_start = 2 / args.num_epochs
    print(f"total_steps: {total_steps}")
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps, final_div_factor=1000, last_epoch=-1, pct_start=pct_start)


def preload_epoch_batches(train_dls, subj_list, voxels, images, num_iterations_per_epoch, batch_size, epoch, args, data_type):
    voxel_iters, perm_iters, betas_iters, select_iters = {}, {}, {}, {}
    image_iters = torch.zeros(num_iterations_per_epoch, batch_size * len(subj_list), 3, 224, 224, dtype=torch.float32)
    mixup_limit = int(args.mixup_pct * args.num_epochs)

    for subj_idx, train_loader in enumerate(train_dls):
        key = f"subj0{subj_list[subj_idx]}"
        batch_idx = -1
        for behav0, _, _, _ in train_loader:
            image_idx = behav0[:, 0, 0].cpu().long().numpy()
            unique_images, image_sorted_idx = np.unique(image_idx, return_index=True)
            if len(unique_images) != len(image_idx):
                continue
            batch_idx += 1
            image_batch = torch.tensor(images[unique_images], dtype=data_type)
            start = subj_idx * batch_size
            image_iters[batch_idx, start : start + batch_size] = image_batch

            voxel_idx = behav0[:, 0, 5].cpu().long().numpy()[image_sorted_idx]
            voxel_batch = voxels[key][voxel_idx].unsqueeze(1)
            if epoch < mixup_limit:
                voxel_batch, perm, betas, select = utils.mixco(voxel_batch)
                perm_iters[f"{key}_iter{batch_idx}"] = perm
                betas_iters[f"{key}_iter{batch_idx}"] = betas
                select_iters[f"{key}_iter{batch_idx}"] = select
            voxel_iters[f"{key}_iter{batch_idx}"] = voxel_batch
            if batch_idx >= num_iterations_per_epoch - 1:
                break
    return image_iters, voxel_iters, perm_iters, betas_iters, select_iters


def maybe_compute_clip_loss(epoch, args, soft_loss_temps, clip_voxels_norm, clip_target_norm, perm=None, betas=None, select=None):
    mixup_limit = int(args.mixup_pct * args.num_epochs)
    if epoch < mixup_limit:
        return utils.mixco_nce(clip_voxels_norm, clip_target_norm, temp=0.006, perm=perm, betas=betas, select=select)
    epoch_temp = soft_loss_temps[epoch - mixup_limit]
    return utils.soft_clip_loss(clip_voxels_norm, clip_target_norm, temp=epoch_temp)

def save_recon_figure(epoch, image, image_enc_pred, autoenc, outdir):
    image = image[:4].float()
    image_enc = autoenc.encode(2 * image - 1).latent_dist.mode().float() * 0.18215
    image_enc_pred = image_enc_pred[:4].float()
    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
    axis_idx = 0
    for image_idx in range(4):
        gt = (autoenc.decode(image_enc[[image_idx]] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
        pred = (autoenc.decode(image_enc_pred[[image_idx]] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
        axes[axis_idx].imshow(utils.torch_to_Image(gt))
        axes[axis_idx].axis("off")
        axis_idx += 1
        axes[axis_idx].imshow(utils.torch_to_Image(pred))
        axes[axis_idx].axis("off")
        axis_idx += 1
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"recon_epoch_{epoch:04d}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curves(outdir, losses, test_losses):
    if not losses and not test_losses:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if losses:
        axes[0].plot(losses)
        axes[0].set_title("train loss")
    if test_losses:
        axes[1].plot(test_losses)
        axes[1].set_title("test loss")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_backbone_norm_curve(outdir, backbone_param_norms, backbone_output_norms, backbone_output_maxs):
    if not backbone_param_norms and not backbone_output_norms and not backbone_output_maxs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if backbone_param_norms:
        axes[0].plot(backbone_param_norms)
        axes[0].set_title("backbone param norm")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("l2 norm")
    if backbone_output_norms:
        axes[1].plot(backbone_output_norms)
        axes[1].set_title("backbone output norm")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("l2 norm")
    if backbone_output_maxs:
        axes[2].plot(backbone_output_maxs)
        axes[2].set_title("backbone output max")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("abs max")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "backbone_norm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_train_parser().parse_args()
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    args.device = torch.device(requested_device)
    if args.outdir is None:
        args.outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train_logs", args.model_name))
    else:
        args.outdir = os.path.abspath(args.outdir)
    utils.seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    data_type = torch.float16 if args.device.type == "cuda" else torch.float32
    autocast_device = args.device.type
    img_augment = make_image_augment() if args.use_image_aug else None

    subj_list = [args.subj]
    if args.multi_subject:
        subj_list = np.arange(1, 9)
        subj_list = subj_list[subj_list != args.subj].tolist()
    print(f"subj_list: {subj_list}, num_sessions: {args.num_sessions}")

    batch_size = args.batch_size // len(subj_list)
    num_samples_per_epoch = 750 * args.num_sessions
    if args.multi_subject:
        num_samples_per_epoch = 750 * 40
    num_iterations_per_epoch = num_samples_per_epoch // (batch_size * len(subj_list))
    print(f"batch_size per subject: {batch_size}, num_iterations_per_epoch: {num_iterations_per_epoch}, num_samples_per_epoch: {num_samples_per_epoch}")

    train_dl, voxels, num_voxels_list, train_urls = load_train_dataloaders(args, subj_list, batch_size, data_type)
    eval_subj = subj_list[0] if args.multi_subject else args.subj
    if args.new_test:
        split = "new_test"
        num_test = {3: 2371, 4: 2188, 6: 2371, 8: 2188}.get(eval_subj, 3000)
    else:
        split = "test"
        num_test = {3: 2113, 4: 1985, 6: 2113, 8: 1985}.get(eval_subj, 2770)
    test_dl, test_url, num_test = load_test_dataloader(args, eval_subj, split, num_test)

    image_h5 = h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r")
    images = image_h5["images"]
    print(f"loaded NSD images: {images.shape}")

    clip_img_embedder = FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", version="laion2b_s39b_b160k", output_tokens=True, only_tokens=True).to(args.device)
    clip_seq_dim, clip_emb_dim = 256, 1664
    recon_modules = build_recon_modules(args)
    model = build_model(args, num_voxels_list, clip_emb_dim, clip_seq_dim)
    optimizer = build_optimizer(model, args)
    lr_scheduler = build_scheduler(optimizer, args, num_iterations_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.type == "cuda")

    print("\nmodel ready")
    num_params = utils.count_params(model)
    print(f"train_urls: {train_urls}")
    print(f"test_url: {test_url}")
    print(f"trainable_params: {num_params}")

    if args.wandb_log:
        import wandb

        wandb_config = vars(args).copy()
        wandb_config.update(
            {
                "num_params": num_params,
                "num_samples_per_epoch": num_samples_per_epoch,
                "num_test": num_test,
                "train_urls": train_urls,
                "test_url": test_url,
            }
        )
        wandb.init(id=args.model_name, project=args.wandb_project, name=args.model_name, config=wandb_config, resume="allow")
    else:
        wandb = None

    epoch = 0
    losses, test_losses, lrs = [], [], []
    backbone_param_norms, backbone_output_norms, backbone_output_maxs = [], [], []
    if args.device.type == "cuda":
        torch.cuda.empty_cache()

    if args.multisubject_ckpt is not None:
        _, _ = load_ckpt("last", model, outdir=args.multisubject_ckpt, strict=False, multisubj_loading=True)

    train_dls = [train_dl[f"subj0{subj}"] for subj in subj_list]
    progress_bar = tqdm(range(epoch, args.num_epochs), ncols=150, desc="Epoch")
    test_image, test_voxel = None, None
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    mixup_limit = int(args.mixup_pct * args.num_epochs)
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - mixup_limit)

    for epoch in progress_bar:
        model.train()
        fwd_percent_correct = bwd_percent_correct = 0.0
        test_fwd_percent_correct = test_bwd_percent_correct = 0.0
        recon_cossim = test_recon_cossim = 0.0
        recon_mse = test_recon_mse = 0.0
        loss_clip_total = loss_blurry_total = loss_blurry_cont_total = 0.0
        test_loss_clip_total = 0.0
        loss_prior_total = test_loss_prior_total = 0.0
        blurry_pixcorr = test_blurry_pixcorr = 0.0
        backbone_output_norm_total = 0.0
        backbone_output_max = 0.0

        image_iters, voxel_iters, perm_iters, betas_iters, select_iters = preload_epoch_batches(
            train_dls, subj_list, voxels, images, num_iterations_per_epoch, batch_size, epoch, args, data_type
        )

        iter_bar = tqdm(range(num_iterations_per_epoch), ncols=150, desc="Iterations", leave=False)
        for train_i in iter_bar:
            optimizer.zero_grad()
            with torch.amp.autocast(autocast_device, dtype=data_type, enabled=args.device.type == "cuda"):
                loss = 0.0
                voxel_list = [voxel_iters[f"subj0{subj}_iter{train_i}"].detach().to(args.device) for subj in subj_list]
                image = image_iters[train_i].detach().to(args.device)
                if img_augment is not None:
                    image = img_augment(image)
                clip_target = clip_img_embedder(image)

                perm = betas = select = None
                if epoch < mixup_limit:
                    perm = torch.cat([perm_iters[f"subj0{subj}_iter{train_i}"].detach().to(args.device) for subj in subj_list], dim=0)
                    betas = torch.cat([betas_iters[f"subj0{subj}_iter{train_i}"].detach().to(args.device) for subj in subj_list], dim=0)
                    select = torch.cat([select_iters[f"subj0{subj}_iter{train_i}"].detach().to(args.device) for subj in subj_list], dim=0)

                voxel_ridge = torch.cat([model.ridge(voxel_list[subj_idx], subj_idx) for subj_idx, _ in enumerate(subj_list)], dim=0)
                backbone, clip_voxels, blurry_image_enc = model.backbone(voxel_ridge)
                backbone_output_norm_total += backbone.float().flatten(1).norm(dim=-1).mean().item()
                backbone_output_max = max(backbone_output_max, backbone.float().abs().max().item())
                clip_target = clip_target.to(dtype=backbone.dtype)

                if args.clip_scale > 0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if args.use_prior:
                    loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                    loss_prior_total += loss_prior.item()
                    loss += loss_prior * args.prior_scale
                    recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                    recon_mse += mse(prior_out, clip_target).item()

                if args.clip_scale > 0:
                    loss_clip = maybe_compute_clip_loss(epoch, args, soft_loss_temps, clip_voxels_norm, clip_target_norm, perm=perm, betas=betas, select=select)
                    loss_clip_total += loss_clip.item()
                    loss += loss_clip * args.clip_scale

                if args.blurry_recon:
                    image_enc_pred, transformer_feats = blurry_image_enc
                    autoenc = recon_modules["autoenc"]
                    cnx = recon_modules["cnx"]
                    mean, std = recon_modules["mean"], recon_modules["std"]
                    blur_augs = recon_modules["blur_augs"]

                    image_enc = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_total += loss_blurry.item()

                    if epoch < mixup_limit:
                        betas_image = betas.to(image_enc.dtype)
                        image_enc_shuf = image_enc[perm]
                        betas_shape = [-1] + [1] * (image_enc.ndim - 1)
                        image_enc[select] = image_enc[select] * betas_image[select].reshape(*betas_shape) + image_enc_shuf[select] * (1 - betas_image[select]).reshape(*betas_shape)

                    image_norm = (image - mean) / std
                    image_aug = (blur_augs(image) - mean) / std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)
                    cont_loss = utils.soft_cont_loss(
                        nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        temp=0.2,
                    )
                    loss_blurry_cont_total += cont_loss.item()
                    loss += (loss_blurry + 0.1 * cont_loss) * args.blur_scale

                if args.clip_scale > 0:
                    labels = torch.arange(len(clip_voxels_norm), device=clip_voxels_norm.device)
                    sims = utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm)
                    fwd_percent_correct += utils.topk(sims, labels, k=1).item()
                    bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                if args.blurry_recon:
                    with torch.no_grad():
                        random_samps = torch.randperm(len(image), device=image.device)[: len(image) // 5]
                        blurry_recon_images = (recon_modules["autoenc"].decode(image_enc_pred[random_samps] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                        blurry_pixcorr += utils.pixcorr(image[random_samps], blurry_recon_images).item()

            utils.check_loss(loss)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None and scaler.get_scale() >= scale_before:
                lr_scheduler.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]["lr"])
            iter_bar.set_postfix(train_loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        iter_bar.close()

        model.eval()
        with torch.no_grad():
            for test_i, (behav, _, _, _) in enumerate(test_dl):
                assert len(behav) == num_test
                if test_image is None:
                    voxel = voxels[f"subj0{eval_subj}"][behav[:, 0, 5].cpu().long()].unsqueeze(1)
                    image_index = behav[:, 0, 0].cpu().long()
                    for im in torch.unique(image_index):
                        locs = torch.where(im == image_index)[0]
                        if len(locs) == 1:
                            locs = locs.repeat(3)
                        elif len(locs) == 2:
                            locs = locs.repeat(2)[:3]
                        image_item = torch.tensor(images[int(im)][None], dtype=torch.float32)
                        if test_image is None:
                            test_image = image_item
                            test_voxel = voxel[locs][None]
                        else:
                            test_image = torch.vstack((test_image, image_item))
                            test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                with torch.amp.autocast(autocast_device, dtype=data_type, enabled=args.device.type == "cuda"):
                    loss = 0.0
                    test_indices = torch.arange(len(test_voxel))[:300]
                    voxel = test_voxel[test_indices].to(args.device)
                    image = test_image[test_indices].to(args.device)
                    clip_target = clip_img_embedder(image.float())

                    clip_voxels = backbone = None
                    for rep in range(3):
                        voxel_ridge = model.ridge(voxel[:, rep], 0)
                        backbone_rep, clip_voxels_rep, blurry_image_enc = model.backbone(voxel_ridge)
                        clip_voxels = clip_voxels_rep if clip_voxels is None else clip_voxels + clip_voxels_rep
                        backbone = backbone_rep if backbone is None else backbone + backbone_rep
                    clip_voxels /= 3
                    backbone /= 3
                    clip_target = clip_target.to(dtype=backbone.dtype)

                    if args.clip_scale > 0:
                        clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                    random_samps = torch.randperm(len(image), device=image.device)[: len(image) // 5]

                    if args.use_prior:
                        loss_prior, prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                        test_loss_prior_total += loss_prior.item()
                        loss += loss_prior * args.prior_scale
                        test_recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target[random_samps]).mean().item()
                        test_recon_mse += mse(prior_out, clip_target[random_samps]).item()

                    if args.clip_scale > 0:
                        loss_clip = utils.soft_clip_loss(clip_voxels_norm, clip_target_norm, temp=0.006)
                        test_loss_clip_total += loss_clip.item()
                        loss += loss_clip * args.clip_scale

                    if args.blurry_recon:
                        image_enc_pred, _ = blurry_image_enc
                        blurry_recon_images = (recon_modules["autoenc"].decode(image_enc_pred[random_samps] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                        test_blurry_pixcorr += utils.pixcorr(image[random_samps], blurry_recon_images).item()

                    if args.clip_scale > 0:
                        labels = torch.arange(len(clip_voxels_norm), device=clip_voxels_norm.device)
                        sims = utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm)
                        test_fwd_percent_correct += utils.topk(sims, labels, k=1).item()
                        test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

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

            backbone_param_norm = 0.0
            for param in model.backbone.parameters():
                backbone_param_norm += param.detach().float().pow(2).sum().item()
            backbone_param_norm = backbone_param_norm ** 0.5
            backbone_output_norm = backbone_output_norm_total / (train_i + 1)
            backbone_param_norms.append(backbone_param_norm)
            backbone_output_norms.append(backbone_output_norm)
            backbone_output_maxs.append(backbone_output_max)

            if args.blurry_recon and ((epoch == args.num_epochs - 1) or (epoch % args.ckpt_interval == 0)):
                save_recon_figure(epoch, image, image_enc_pred, recon_modules["autoenc"], args.outdir)
            progress_bar.set_postfix(
                lr=f"{logs['train/lr']:.2e}",
                tr=f"{logs['train/loss']:.3f}",
                te=f"{logs['test/loss']:.3f}",
            )
            progress_bar.write(
                f"epoch {epoch + 1}/{args.num_epochs} "
                f"train={logs['train/loss']:.4f} test={logs['test/loss']:.4f} "
                f"clip={logs['train/loss_clip_total']:.4f} prior={logs['train/loss_prior']:.4f} "
                f"pixcorr={logs['test/blurry_pixcorr']:.4f} bb_param_norm={backbone_param_norm:.1f} "
                f"bb_output_norm={backbone_output_norm:.1f} bb_output_max={backbone_output_max:.1f} "
                f"train_fwd_pct={logs['train/fwd_pct_correct']:.3f} "
                f"train_bwd_pct={logs['train/bwd_pct_correct']:.3f} "
                f"test_fwd_pct={logs['test/test_fwd_pct_correct']:.3f} "
                f"test_bwd_pct={logs['test/test_bwd_pct_correct']:.3f}"
            )
            if wandb is not None:
                wandb.log(logs)

        if args.ckpt_saving and epoch % args.ckpt_interval == 0:
            save_ckpt("last", model, optimizer, epoch, lr_scheduler, losses, test_losses, lrs, args.outdir)
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nfinished training\n")
    if args.ckpt_saving:
        save_ckpt("last", model, optimizer, epoch, lr_scheduler, losses, test_losses, lrs, args.outdir)
    save_loss_curves(args.outdir, losses, test_losses)
    save_backbone_norm_curve(args.outdir, backbone_param_norms, backbone_output_norms, backbone_output_maxs)
    image_h5.close()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

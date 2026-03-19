import os
import sys
import argparse
from typing import Tuple

import h5py
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

import utils
from models.mindeye_prior import PriorNetwork, BrainDiffusionPrior
from models.model import FMRIModel

# SDXL unCLIP
sys.path.append("generative_models/")
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf


torch.backends.cuda.matmul.allow_tf32 = True


def build_test_url(data_path: str, subj: int, new_test: bool) -> Tuple[str, int]:
    if new_test:
        n = {1: 3000, 2: 3000, 3: 2371, 4: 2188, 5: 3000, 6: 2371, 7: 3000, 8: 2188}[subj]
        return f"{data_path}/wds/subj0{subj}/new_test/0.tar", n
    n = {1: 2770, 2: 2770, 3: 2113, 4: 1985, 5: 2770, 6: 2113, 7: 2770, 8: 1985}[subj]
    return f"{data_path}/wds/subj0{subj}/test/0.tar", n


def load_subject_voxels(data_path: str, subj: int) -> torch.Tensor:
    with h5py.File(f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
        return torch.tensor(f["betas"][:], dtype=torch.float32)


def build_test_loader(test_url: str, batch_size: int) -> torch.utils.data.DataLoader:
    test_data = (
        wds.WebDataset(test_url, resampled=False)
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    return torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


def build_fmri_model(args, voxel_dim: int, device: torch.device) -> FMRIModel:
    model = FMRIModel(
        voxel_dims=[voxel_dim],
        hidden_dim=args.fmri_hidden_dim,
        fmri_feature_dim=args.fmri_feature_dim,
        fmri_feature_seq_len=args.fmri_feature_seq_len,
        timestep=args.fmri_timestep,
        ae_enc_depth=args.fmri_ae_enc_depth,
        ae_dec_depth=args.fmri_ae_dec_depth,
    ).to(device)
    return model



def load_fmri_encoder(args, voxel_dim: int, device: torch.device) -> FMRIModel:
    model = build_fmri_model(args, voxel_dim=voxel_dim, device=device)
    ckpt = torch.load(args.fmri_ckpt, map_location="cpu")
    state = ckpt.get("encoder", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded FMRI ckpt: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval().requires_grad_(False)
    return model


def extract_cond_tokens(features) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        cond = features
    elif isinstance(features, dict):
        if "cond_tokens" in features:
            cond = features["cond_tokens"]
        elif "middle_feature" in features:
            cond = features["middle_feature"]
        elif "features" in features:
            cond = features["features"]
        else:
            raise KeyError(f"Unsupported FMRI feature keys: {list(features.keys())}")
    else:
        raise TypeError(f"Unsupported FMRI feature type: {type(features)}")

    if cond.dim() == 2:
        cond = cond.unsqueeze(1)
    return cond.detach().float()


def load_model_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for key in ("diffusion_prior",):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def load_diffusion_model(args, device: torch.device) -> BrainDiffusionPrior:
    prior_network = PriorNetwork(
        dim=args.clip_emb_dim,
        depth=args.prior_depth,
        dim_head=args.dim_head,
        heads=args.clip_emb_dim // args.dim_head,
        causal=False,
        num_tokens=args.clip_seq_dim,
        learned_query_mode="pos_emb",
    )

    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=args.clip_emb_dim,
        condition_on_text_encodings=False,
        timesteps=args.T,
        cond_drop_prob=args.cond_drop_prob,
        image_embed_scale=None,
    ).to(device)

    state = load_model_state(args.diffusion_ckpt)
    missing, unexpected = diffusion_prior.load_state_dict(state, strict=False)
    print(f"Loaded diffusion_prior ckpt: missing={len(missing)}, unexpected={len(unexpected)}")
    diffusion_prior.eval().requires_grad_(False)
    return diffusion_prior


def load_unclip_engine(cache_dir: str, device: torch.device) -> Tuple[DiffusionEngine, torch.Tensor]:
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)

    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]

    first_stage_config["target"] = "sgm.models.autoencoder.AutoencoderKL"
    sampler_config["params"]["num_steps"] = 38

    engine = DiffusionEngine(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        conditioner_config=conditioner_config,
        sampler_config=sampler_config,
        scale_factor=scale_factor,
        disable_first_stage_autocast=disable_first_stage_autocast,
    )
    engine.to(device)
    engine.eval().requires_grad_(False)

    ckpt = torch.load(f"{cache_dir}/unclip6_epoch0_step110000.ckpt", map_location="cpu")
    engine.load_state_dict(ckpt["state_dict"])

    batch = {
        "jpg": torch.randn(1, 3, 1, 1, device=device),
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
    }
    out = engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    return engine, vector_suffix


def collect_unique_test_pairs(
    test_loader: torch.utils.data.DataLoader,
    subj_voxels: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    image_indices = []
    voxel_indices = []

    for behav, _, _, _ in test_loader:
        image_indices.append(behav[:, 0, 0].cpu().numpy())
        voxel_indices.append(behav[:, 0, 5].cpu().numpy())

    image_indices = np.concatenate(image_indices).astype(int)
    voxel_indices = np.concatenate(voxel_indices).astype(int)

    unique_imgs = np.unique(image_indices)
    averaged_voxels = []

    for img_id in unique_imgs:
        locs = np.where(image_indices == img_id)[0]
        if len(locs) == 1:
            locs = np.repeat(locs, 3)
        elif len(locs) == 2:
            locs = np.tile(locs, 2)[:3]
        else:
            locs = locs[:3]

        vox = subj_voxels[voxel_indices[locs]]
        averaged_voxels.append(vox.mean(dim=0))

    averaged_voxels = torch.stack(averaged_voxels, dim=0)
    return averaged_voxels, unique_imgs, torch.tensor(image_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct images: voxel -> fmri encoder -> diffusion -> unCLIP")
    parser.add_argument("--model_name", type=str, default="ffl_mse")
    parser.add_argument("--data_path", type=str, default="/data20TB/lzg/MindEyeV2")
    parser.add_argument("--cache_dir", type=str, default="/data20TB/lzg/MindEyeV2")
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--fmri_ckpt", type=str, default="/data20TB/lzg/MindEyeV2/train_logs/fmri_encoder/final/best.pth")
    parser.add_argument("--fmri_hidden_dim", type=int, default=4096)
    parser.add_argument("--fmri_feature_dim", type=int, default=1664)
    parser.add_argument("--fmri_feature_seq_len", type=int, default=256)
    parser.add_argument("--fmri_timestep", type=int, default=4)
    parser.add_argument("--fmri_ae_enc_depth", type=int, default=4)
    parser.add_argument("--fmri_ae_dec_depth", type=int, default=4)

    # Old FMRIModel args for compatibility fallback.
    parser.add_argument("--fmri_latent_dim", type=int, default=512)
    parser.add_argument("--fmri_seq_len", type=int, default=32)
    parser.add_argument("--fmri_n_heads", type=int, default=8)
    parser.add_argument("--fmri_ar_depth", type=int, default=2)
    parser.add_argument("--fmri_bottleneck_dim", type=int, default=256)

    parser.add_argument("--diffusion_ckpt", type=str, required=True)

    parser.add_argument("--clip_seq_dim", type=int, default=256)
    parser.add_argument("--clip_emb_dim", type=int, default=1664)
    parser.add_argument("--T", type=int, default=100)

    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--prior_depth", type=int, default=6)
    parser.add_argument("--dim_head", type=int, default=52)
    parser.add_argument("--cond_drop_prob", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples_per_image", type=int, default=1)

    args = parser.parse_args()

    utils.seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    subj_voxels = load_subject_voxels(args.data_path, args.subj)
    print(f"num_voxels for subj0{args.subj}: {subj_voxels.shape[-1]}")

    test_url, num_test = build_test_url(args.data_path, args.subj, args.new_test)
    print(f"test_url: {test_url}")
    test_loader = build_test_loader(test_url, batch_size=num_test)

    with h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r") as f:
        all_images_h5 = f["images"][:]
    print(f"Loaded NSD images: {all_images_h5.shape}")

    fmri_encoder = load_fmri_encoder(args, voxel_dim=subj_voxels.shape[-1], device=device)
    diffusion_prior = load_diffusion_model(args, device=device)
    unclip_engine, vector_suffix = load_unclip_engine(args.cache_dir, device)

    averaged_voxels, unique_imgs, _ = collect_unique_test_pairs(test_loader, subj_voxels)
    print(f"Unique test images: {len(unique_imgs)}")

    all_recons = []
    all_clipemb = []

    for start in tqdm(range(0, len(unique_imgs), args.batch_size), desc="Reconstruct", ncols=120):
        end = min(start + args.batch_size, len(unique_imgs))

        voxel_batch = averaged_voxels[start:end].unsqueeze(1).to(device)
        with torch.no_grad():
            _, feats = fmri_encoder(voxel_batch, 0, returnFeatures=True)
            cond_tokens = extract_cond_tokens(feats)
            clip_pred = diffusion_prior.p_sample_loop(
                cond_tokens.shape,
                text_cond=dict(text_embed=cond_tokens),
                cond_scale=1.0,
            )

        all_clipemb.append(clip_pred.cpu())

        for i in range(clip_pred.shape[0]):
            samples = utils.unclip_recon(
                clip_pred[[i]],
                unclip_engine,
                vector_suffix,
                num_samples=args.num_samples_per_image,
            )
            all_recons.append(samples.cpu())

    all_recons = torch.cat(all_recons, dim=0).float()
    all_clipemb = torch.cat(all_clipemb, dim=0).float()
    all_images = torch.tensor(all_images_h5[unique_imgs], dtype=torch.float32)

    outdir = os.path.abspath(f"evals/{args.model_name}")
    os.makedirs(outdir, exist_ok=True)

    torch.save(all_recons, os.path.join(outdir, f"{args.model_name}_all_recons.pt"))
    torch.save(all_images, os.path.join(outdir, "all_images.pt"))
    torch.save(torch.tensor(unique_imgs), os.path.join(outdir, "all_image_indices.pt"))
    torch.save(all_clipemb, os.path.join(outdir, f"{args.model_name}_all_clipemb.pt"))

    print(f"Saved recon images and clip embeddings to {outdir}")


if __name__ == "__main__":
    main()

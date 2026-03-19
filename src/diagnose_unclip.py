import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import webdataset as wds
from torchvision import transforms

from models.fft_diffusion import FFTBrainDiffusionPrior
from models.model import FMRIModel

try:
    from diffusion_train import CondDiffusionDenoiser  # type: ignore
except ImportError:
    CondDiffusionDenoiser = None

sys.path.append("generative_models/")
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from omegaconf import OmegaConf


torch.backends.cuda.matmul.allow_tf32 = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    while x.ndim < target_dims:
        x = x.unsqueeze(-1)
    return x


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_bar: torch.Tensor


class ConditionalDDPMSampler:
    def __init__(self, model: torch.nn.Module, T: int, beta_1: float, beta_T: float, device: torch.device):
        self.model = model
        self.T = T
        betas = torch.linspace(beta_1, beta_T, T, device=device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.schedule = DiffusionSchedule(betas=betas, alphas=alphas, alphas_bar=alphas_bar)

    @torch.no_grad()
    def sample(self, cond_tokens: torch.Tensor, out_shape: Tuple[int, ...]) -> torch.Tensor:
        x_t = torch.randn(*out_shape, device=cond_tokens.device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((out_shape[0],), t, device=cond_tokens.device, dtype=torch.long)
            eps = self.model(x_t, t_batch, cond_tokens)
            beta_t = self.schedule.betas[t]
            alpha_t = self.schedule.alphas[t]
            alpha_bar_t = self.schedule.alphas_bar[t]
            coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mean = (x_t - coef * eps) / torch.sqrt(alpha_t)
            if t > 0:
                x_t = mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
            else:
                x_t = mean
        return x_t


def build_test_url(data_path: str, subj: int, new_test: bool) -> Tuple[str, int]:
    if new_test:
        n = {1: 3000, 2: 3000, 3: 2371, 4: 2188, 5: 3000, 6: 2371, 7: 3000, 8: 2188}[subj]
        return f"{data_path}/wds/subj0{subj}/new_test/0.tar", n
    n = {1: 2770, 2: 2770, 3: 2113, 4: 1985, 5: 2770, 6: 2113, 7: 2770, 8: 1985}[subj]
    return f"{data_path}/wds/subj0{subj}/test/0.tar", n


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
    return torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)


def collect_unique_test_pairs(
    test_loader: torch.utils.data.DataLoader,
    subj_voxels: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray]:
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
        averaged_voxels.append(subj_voxels[voxel_indices[locs]].mean(dim=0))
    return torch.stack(averaged_voxels, dim=0), unique_imgs


def build_fmri_model(args, voxel_dim: int, device: torch.device):
    try:
        return FMRIModel(
            voxel_dims=[voxel_dim],
            hidden_dim=args.fmri_hidden_dim,
            fmri_feature_dim=args.fmri_feature_dim,
            fmri_feature_seq_len=args.fmri_feature_seq_len,
            timestep=args.fmri_timestep,
            ae_enc_depth=args.fmri_ae_enc_depth,
            ae_dec_depth=args.fmri_ae_dec_depth,
        ).to(device)
    except TypeError:
        return FMRIModel(
            voxel_dims=[voxel_dim],
            latent_dim=args.fmri_latent_dim,
            seq_len=args.fmri_seq_len,
            timestep=args.fmri_timestep,
            n_heads=args.fmri_n_heads,
            ar_depth=args.fmri_ar_depth,
            ae_enc_depth=args.fmri_ae_enc_depth,
            ae_dec_depth=args.fmri_ae_dec_depth,
            bottleneck_dim=args.fmri_bottleneck_dim,
        ).to(device)


def load_fmri_encoder(args, voxel_dim: int, device: torch.device):
    model = build_fmri_model(args, voxel_dim, device)
    ckpt = torch.load(args.fmri_ckpt, map_location="cpu")
    state = ckpt.get("encoder", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded FMRI ckpt: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval().requires_grad_(False)
    return model


def extract_cond_tokens(features) -> torch.Tensor:
    cond = features["middle_feature"] if isinstance(features, dict) and "middle_feature" in features else features
    if cond.dim() == 2:
        cond = cond.unsqueeze(1)
    return cond.detach().float()


def load_model_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for key in ("net_model", "model", "model_state_dict"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def load_fft_diffusion_model(args, device: torch.device) -> FFTBrainDiffusionPrior:
    denoiser = FFTBrainDiffusionPrior(
        clip_emb_dim=args.clip_emb_dim,
        num_tokens=args.clip_seq_dim,
        num_timesteps=args.T,
        depth=args.prior_depth,
        heads=args.prior_heads,
        dropout=args.dropout,
        causal=args.prior_causal,
        learned_query_mode=args.prior_learned_query_mode,
        cond_drop_prob=args.prior_cond_drop_prob,
        out_channels=2,
    ).to(device)
    state = load_model_state(args.diffusion_ckpt)
    missing, unexpected = denoiser.load_state_dict(state, strict=False)
    print(f"Loaded FFT diffusion ckpt: missing={len(missing)}, unexpected={len(unexpected)}")
    denoiser.eval().requires_grad_(False)
    return denoiser


def load_clip_diffusion_model(args, device: torch.device) -> torch.nn.Module:
    if CondDiffusionDenoiser is None:
        raise ImportError("CondDiffusionDenoiser is unavailable; use --diffusion_type fft.")
    denoiser = CondDiffusionDenoiser(
        clip_emb_dim=args.clip_emb_dim,
        cond_dim=args.fmri_latent_dim,
        hidden_dim=args.diff_hidden_dim,
        n_heads=args.diff_heads,
        depth=args.diff_depth,
        T=args.T,
    ).to(device)
    state = load_model_state(args.diffusion_ckpt)
    missing, unexpected = denoiser.load_state_dict(state, strict=False)
    print(f"Loaded clip diffusion ckpt: missing={len(missing)}, unexpected={len(unexpected)}")
    denoiser.eval().requires_grad_(False)
    return denoiser


def load_diffusion_model(args, device: torch.device) -> Tuple[torch.nn.Module, str]:
    if args.diffusion_type == "fft":
        return load_fft_diffusion_model(args, device), "fft"
    if args.diffusion_type == "clip":
        return load_clip_diffusion_model(args, device), "clip"
    try:
        return load_fft_diffusion_model(args, device), "fft"
    except Exception as fft_error:
        print(f"FFT diffusion load failed, falling back to clip diffusion: {fft_error}")
        return load_clip_diffusion_model(args, device), "clip"


def fft_channels_to_clip_tokens(x_fft: torch.Tensor, mode: str = "ri") -> torch.Tensor:
    if mode == "ri":
        spectrum = torch.complex(x_fft[..., 0], x_fft[..., 1])
    elif mode == "mp":
        spectrum = torch.polar(torch.expm1(x_fft[..., 0]).clamp_min(0.0), x_fft[..., 1])
    else:
        raise ValueError(f"Unknown fft_mode={mode}")
    return torch.fft.ifft(spectrum, dim=1, norm="ortho").real.float()


def clip_to_fft_channels(clip_tokens: torch.Tensor, mode: str = "ri") -> torch.Tensor:
    spec = torch.fft.fft(clip_tokens, dim=1, norm="ortho")
    if mode == "ri":
        return torch.stack([spec.real, spec.imag], dim=-1)
    if mode == "mp":
        return torch.stack([torch.log1p(torch.abs(spec)), torch.angle(spec)], dim=-1)
    raise ValueError(f"Unknown fft_mode={mode}")


def load_unclip_engine(cache_dir: str, device: torch.device) -> Tuple[DiffusionEngine, torch.Tensor]:
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    params = config["model"]["params"]
    params["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"
    params["sampler_config"]["params"]["num_steps"] = 38
    engine = DiffusionEngine(
        network_config=params["network_config"],
        denoiser_config=params["denoiser_config"],
        first_stage_config=params["first_stage_config"],
        conditioner_config=params["conditioner_config"],
        sampler_config=params["sampler_config"],
        scale_factor=params["scale_factor"],
        disable_first_stage_autocast=params["disable_first_stage_autocast"],
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
    vector_suffix = engine.conditioner(batch)["vector"].to(device)
    return engine, vector_suffix


@torch.no_grad()
def unclip_recon(x: torch.Tensor, diffusion_engine: DiffusionEngine, vector_suffix: torch.Tensor, num_samples: int = 1, offset_noise_level: float = 0.04) -> torch.Tensor:
    z = torch.randn(num_samples, 4, 96, 96, device=x.device)
    c = {"crossattn": x.repeat(num_samples, 1, 1), "vector": vector_suffix.repeat(num_samples, 1)}
    uc = {"crossattn": torch.randn_like(x).repeat(num_samples, 1, 1), "vector": vector_suffix.repeat(num_samples, 1)}
    sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
    sigma = sigmas[0].to(z.device)
    noise = torch.randn_like(z)
    if offset_noise_level > 0.0:
        noise = noise + offset_noise_level * append_dims(torch.randn(z.shape[0], device=z.device), z.ndim)
    noised_z = (z + noise * append_dims(sigma, z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)

    def denoiser(x_in, sigma_in, cond_in):
        return diffusion_engine.denoiser(diffusion_engine.model, x_in, sigma_in, cond_in)

    with torch.amp.autocast("cuda", dtype=torch.float16), diffusion_engine.ema_scope():
        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
    return torch.clamp(samples_x * 0.8 + 0.2, 0.0, 1.0)


def cosine_diag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.flatten(1), dim=-1)
    b = torch.nn.functional.normalize(b.flatten(1), dim=-1)
    return (a * b).sum(dim=-1)


def retrieval_at_1(query: torch.Tensor, target: torch.Tensor) -> float:
    query = torch.nn.functional.normalize(query.flatten(1), dim=-1)
    target = torch.nn.functional.normalize(target.flatten(1), dim=-1)
    sims = query @ target.T
    preds = sims.argmax(dim=1)
    labels = torch.arange(len(query), device=query.device)
    return float((preds == labels).float().mean().item())


def summarize_tokens(name: str, x: torch.Tensor) -> None:
    flat = x.flatten(1)
    norms = flat.norm(dim=1)
    print(
        f"{name}: shape={tuple(x.shape)} "
        f"mean={x.mean().item():.4f} std={x.std().item():.4f} "
        f"min={x.min().item():.4f} max={x.max().item():.4f} "
        f"norm_mean={norms.mean().item():.4f} norm_std={norms.std().item():.4f}"
    )


def save_triptych(images: torch.Tensor, gt_recons: torch.Tensor, pred_recons: torch.Tensor, output_path: str, resize: int) -> None:
    resize_tf = transforms.Resize((resize, resize))
    images = resize_tf(images).cpu()
    gt_recons = resize_tf(gt_recons).cpu()
    pred_recons = resize_tf(pred_recons).cpu()

    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(9, max(3, n * 3)))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(gt_recons[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("GT clip -> unCLIP")
        axes[i, 2].imshow(pred_recons[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title("Pred clip -> unCLIP")
        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether noisy reconstructions come from prior tokens or unCLIP decoding.")
    parser.add_argument("--data_path", type=str, default="/data20TB/lzg/MindEyeV2")
    parser.add_argument("--cache_dir", type=str, default="/data20TB/lzg/MindEyeV2")
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="evals/diagnostics")

    parser.add_argument("--fmri_ckpt", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--diffusion_type", type=str, default="fft", choices=["fft", "clip", "auto"])
    parser.add_argument("--fft_mode", type=str, default="ri", choices=["ri", "mp"])

    parser.add_argument("--fmri_hidden_dim", type=int, default=4096)
    parser.add_argument("--fmri_feature_dim", type=int, default=1664)
    parser.add_argument("--fmri_feature_seq_len", type=int, default=256)
    parser.add_argument("--fmri_timestep", type=int, default=4)
    parser.add_argument("--fmri_ae_enc_depth", type=int, default=2)
    parser.add_argument("--fmri_ae_dec_depth", type=int, default=2)
    parser.add_argument("--fmri_latent_dim", type=int, default=512)
    parser.add_argument("--fmri_seq_len", type=int, default=32)
    parser.add_argument("--fmri_n_heads", type=int, default=8)
    parser.add_argument("--fmri_ar_depth", type=int, default=2)
    parser.add_argument("--fmri_bottleneck_dim", type=int, default=256)

    parser.add_argument("--clip_seq_dim", type=int, default=256)
    parser.add_argument("--clip_emb_dim", type=int, default=1664)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--prior_depth", type=int, default=6)
    parser.add_argument("--prior_heads", type=int, default=8)
    parser.add_argument("--prior_causal", action="store_true")
    parser.add_argument("--prior_cond_drop_prob", type=float, default=0.0)
    parser.add_argument("--prior_learned_query_mode", type=str, default="pos_emb", choices=["none", "token", "pos_emb", "all_pos_emb"])

    parser.add_argument("--diff_hidden_dim", type=int, default=1664)
    parser.add_argument("--diff_heads", type=int, default=8)
    parser.add_argument("--diff_depth", type=int, default=4)

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    with h5py.File(f"{args.data_path}/betas_all_subj0{args.subj}_fp32_renorm.hdf5", "r") as f:
        subj_voxels = torch.tensor(f["betas"][:], dtype=torch.float32)
    with h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r") as f:
        all_images_h5 = f["images"][:]

    test_url, num_test = build_test_url(args.data_path, args.subj, args.new_test)
    test_loader = build_test_loader(test_url, batch_size=num_test)
    averaged_voxels, unique_imgs = collect_unique_test_pairs(test_loader, subj_voxels)
    num_examples = min(args.num_examples, len(unique_imgs))
    averaged_voxels = averaged_voxels[:num_examples].unsqueeze(1).to(device)
    images = torch.tensor(all_images_h5[unique_imgs[:num_examples]], dtype=torch.float32, device=device)

    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    ).to(device)
    clip_img_embedder.eval().requires_grad_(False)

    fmri_encoder = load_fmri_encoder(args, voxel_dim=subj_voxels.shape[-1], device=device)
    denoiser, diffusion_type = load_diffusion_model(args, device=device)
    sampler = ConditionalDDPMSampler(denoiser, T=args.T, beta_1=args.beta_1, beta_T=args.beta_T, device=device)
    unclip_engine, vector_suffix = load_unclip_engine(args.cache_dir, device)

    with torch.no_grad():
        gt_clip = clip_img_embedder(images).float()
        _, feats = fmri_encoder(averaged_voxels, 0, returnFeatures=True)
        cond_tokens = extract_cond_tokens(feats)
        if diffusion_type == "fft":
            pred_fft = sampler.sample(cond_tokens, (num_examples, args.clip_seq_dim, args.clip_emb_dim, 2))
            pred_clip = fft_channels_to_clip_tokens(pred_fft, mode=args.fft_mode)
            gt_fft = clip_to_fft_channels(gt_clip, mode=args.fft_mode)
            gt_roundtrip = fft_channels_to_clip_tokens(gt_fft, mode=args.fft_mode)
        else:
            pred_clip = sampler.sample(cond_tokens, (num_examples, args.clip_seq_dim, args.clip_emb_dim))
            gt_roundtrip = gt_clip

    summarize_tokens("gt_clip", gt_clip)
    summarize_tokens("pred_clip", pred_clip)
    summarize_tokens("pred_minus_gt", pred_clip - gt_clip)
    print(f"diag cosine mean={cosine_diag(pred_clip, gt_clip).mean().item():.4f}")
    print(f"retrieval@1={retrieval_at_1(pred_clip, gt_clip):.4f}")
    if diffusion_type == "fft":
        print(f"fft roundtrip cosine mean={cosine_diag(gt_roundtrip, gt_clip).mean().item():.4f}")

    gt_recons = []
    pred_recons = []
    for i in range(num_examples):
        gt_recons.append(unclip_recon(gt_clip[[i]], unclip_engine, vector_suffix, num_samples=1).cpu())
        pred_recons.append(unclip_recon(pred_clip[[i]], unclip_engine, vector_suffix, num_samples=1).cpu())

    gt_recons = torch.cat(gt_recons, dim=0)
    pred_recons = torch.cat(pred_recons, dim=0)

    output_path = os.path.join(args.output_dir, f"subj{args.subj:02d}_{diffusion_type}_diagnosis.png")
    save_triptych(images.cpu(), gt_recons, pred_recons, output_path, resize=args.resize)


if __name__ == "__main__":
    main()

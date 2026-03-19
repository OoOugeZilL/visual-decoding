import os
import argparse
from typing import Callable, Dict, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import webdataset as wds
from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights,
    Inception_V3_Weights,
    EfficientNet_B1_Weights,
    alexnet,
    inception_v3,
    efficientnet_b1,
)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True


def safe_run(fn: Callable[[], float], name: str) -> float:
    try:
        return float(fn())
    except Exception as e:
        print(f"[WARN] {name} failed: {e}")
        return float("nan")


def safe_run_obj(fn: Callable, name: str, fallback):
    try:
        return fn()
    except Exception as e:
        print(f"[WARN] {name} failed: {e}")
        return fallback


def resolve_existing_path(*candidates: str) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def build_clip_image_embedder(device: torch.device):
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

    embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    ).to(device)
    embedder.eval().requires_grad_(False)
    return embedder


def batchwise_pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_centered = a - a.mean(dim=1, keepdim=True)
    b_centered = b - b.mean(dim=1, keepdim=True)
    numerator = a_centered @ b_centered.T
    denominator = torch.linalg.norm(a_centered, dim=1, keepdim=True) @ torch.linalg.norm(b_centered, dim=1, keepdim=True).T
    return numerator / denominator.clamp_min(1e-8)


def batchwise_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten(1)
    b = b.flatten(1).T
    a_norm = torch.linalg.norm(a, dim=1, keepdim=True)
    b_norm = torch.linalg.norm(b, dim=0, keepdim=True)
    return ((a @ b) / (a_norm @ b_norm).clamp_min(1e-8)).T


def topk(similarities: torch.Tensor, labels: torch.Tensor, k: int = 5) -> torch.Tensor:
    k = min(k, similarities.shape[0])
    topsum = 0.0
    for i in range(k):
        preds = torch.argsort(similarities, dim=1)[:, -(i + 1)]
        topsum += torch.sum(preds == labels) / len(labels)
    return topsum


def compute_pixcorr(images: torch.Tensor, recons: torch.Tensor, nan: bool = True) -> torch.Tensor:
    resize_425 = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    images_flat = resize_425(images).reshape(len(images), -1)
    recons_flat = resize_425(recons).reshape(len(recons), -1)
    corr = torch.diag(batchwise_pearson_correlation(images_flat, recons_flat))
    return torch.nanmean(corr) if nan else torch.mean(corr)


def batch_pearson_diag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [N,D]
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-8)
    return (a * b).sum(dim=1) / denom


@torch.no_grad()
def two_way_identification(
    recons: torch.Tensor,
    images: torch.Tensor,
    model_fn: Callable,
    preprocess: Callable,
    feature_key: str = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    x_pred = torch.stack([preprocess(x) for x in recons], dim=0).to(device)
    x_gt = torch.stack([preprocess(x) for x in images], dim=0).to(device)

    pred_feats = model_fn(x_pred)
    gt_feats = model_fn(x_gt)

    if feature_key is not None:
        pred_feats = pred_feats[feature_key]
        gt_feats = gt_feats[feature_key]

    pred_feats = pred_feats.float().flatten(1).cpu().numpy()
    gt_feats = gt_feats.float().flatten(1).cpu().numpy()

    r = np.corrcoef(gt_feats, pred_feats)
    r = r[: len(gt_feats), len(gt_feats) :]
    congruent = np.diag(r)
    success = r < congruent
    success_cnt = np.sum(success, axis=0)
    return float(np.mean(success_cnt) / (len(gt_feats) - 1))


@torch.no_grad()
def clip_fwd_bwd_retrieval(
    all_images: torch.Tensor,
    all_clipemb: torch.Tensor,
    device: torch.device,
    n_loops: int = 30,
    sample_size: int = 300,
) -> Tuple[float, float]:
    clip_img_embedder = build_clip_image_embedder(device)

    fwd_scores, bwd_scores = [], []
    n = len(all_images)

    for _ in tqdm(range(n_loops), desc="Retrieval", ncols=120):
        idx = np.random.choice(np.arange(n), size=min(sample_size, n), replace=False)

        emb_img = clip_img_embedder(all_images[idx].to(device)).float().reshape(len(idx), -1)
        emb_brain = all_clipemb[idx].to(device).float().reshape(len(idx), -1)

        emb_img = F.normalize(emb_img, dim=-1)
        emb_brain = F.normalize(emb_brain, dim=-1)

        labels = torch.arange(len(idx), device=device)
        bwd_sim = batchwise_cosine_similarity(emb_img, emb_brain)
        fwd_sim = batchwise_cosine_similarity(emb_brain, emb_img)

        fwd_scores.append(topk(fwd_sim, labels, k=1).item())
        bwd_scores.append(topk(bwd_sim, labels, k=1).item())

    return float(np.mean(fwd_scores)), float(np.mean(bwd_scores))


def build_test_url(data_path: str, subj: int, new_test: bool) -> Tuple[str, int]:
    if new_test:
        n = {1: 3000, 2: 3000, 3: 2371, 4: 2188, 5: 3000, 6: 2371, 7: 3000, 8: 2188}[subj]
        return f"{data_path}/wds/subj0{subj}/new_test/0.tar", n
    n = {1: 2770, 2: 2770, 3: 2113, 4: 1985, 5: 2770, 6: 2113, 7: 2770, 8: 1985}[subj]
    return f"{data_path}/wds/subj0{subj}/test/0.tar", n


def build_fmri_model(args, voxel_dim: int, device: torch.device):
    from models.model import FMRIModel

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


@torch.no_grad()
def brain_region_correlation(
    args,
    device: torch.device,
) -> Dict[str, float]:
    with h5py.File(f"{args.data_path}/betas_all_subj0{args.subj}_fp32_renorm.hdf5", "r") as f:
        subj_voxels = torch.tensor(f["betas"][:], dtype=torch.float32)

    test_url, num_test = build_test_url(args.data_path, args.subj, args.new_test)
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
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=False, pin_memory=True)

    image_idx_all, voxel_idx_all = [], []
    for behav, _, _, _ in test_dl:
        image_idx_all.append(behav[:, 0, 0].cpu().numpy())
        voxel_idx_all.append(behav[:, 0, 5].cpu().numpy())

    image_idx_all = np.concatenate(image_idx_all).astype(int)
    voxel_idx_all = np.concatenate(voxel_idx_all).astype(int)

    unique_imgs = np.unique(image_idx_all)
    gt_avg_voxels = []

    for img_id in unique_imgs:
        locs = np.where(image_idx_all == img_id)[0]
        if len(locs) == 1:
            locs = np.repeat(locs, 3)
        elif len(locs) == 2:
            locs = np.tile(locs, 2)[:3]
        else:
            locs = locs[:3]
        gt_avg_voxels.append(subj_voxels[voxel_idx_all[locs]].mean(dim=0))

    gt_avg_voxels = torch.stack(gt_avg_voxels, dim=0).to(device)  # [N,V]

    model = build_fmri_model(args, voxel_dim=gt_avg_voxels.shape[-1], device=device)

    ckpt = torch.load(args.fmri_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt.get("encoder", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval().requires_grad_(False)

    pred_avg_voxels = model(gt_avg_voxels.unsqueeze(1), 0).squeeze(1)

    with h5py.File(f"{args.data_path}/brain_region_masks.hdf5", "r") as f:
        subj_group = f[f"subj0{args.subj}"]
        masks = {
            "nsd_general": subj_group["nsd_general"][:],
            "V1": subj_group["V1"][:],
            "V2": subj_group["V2"][:],
            "V3": subj_group["V3"][:],
            "V4": subj_group["V4"][:],
            "higher_vis": subj_group["higher_vis"][:],
        }

    out = {}
    for region, mask in masks.items():
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        a = gt_avg_voxels[:, mask_t].transpose(0, 1)
        b = pred_avg_voxels[:, mask_t].transpose(0, 1)
        corr_per_voxel = batch_pearson_diag(a, b)
        out[f"Brain Corr. {region}"] = float(torch.mean(corr_per_voxel).item())

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Full image reconstruction evaluation")
    parser.add_argument("--model_name", type=str, default="fmri_diffusion_recon")
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="/data20TB/lzg/MindEyeV2")
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--fmri_ckpt", type=str, default="/data20TB/lzg/MindEyeV2/train_logs/fmri_encoder/final/best.pth")
    parser.add_argument("--fmri_hidden_dim", type=int, default=4096)
    parser.add_argument("--fmri_feature_dim", type=int, default=1664)
    parser.add_argument("--fmri_feature_seq_len", type=int, default=256)
    parser.add_argument("--fmri_latent_dim", type=int, default=512)
    parser.add_argument("--fmri_seq_len", type=int, default=32)
    parser.add_argument("--fmri_timestep", type=int, default=4)
    parser.add_argument("--fmri_n_heads", type=int, default=8)
    parser.add_argument("--fmri_ar_depth", type=int, default=2)
    parser.add_argument("--fmri_ae_enc_depth", type=int, default=4)
    parser.add_argument("--fmri_ae_dec_depth", type=int, default=4)
    parser.add_argument("--fmri_bottleneck_dim", type=int, default=256)

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    eval_dir = args.eval_dir or os.path.abspath(f"evals/{args.model_name}")
    recon_path = os.path.join(eval_dir, f"{args.model_name}_all_recons.pt")
    gt_path = resolve_existing_path(
        os.path.join(eval_dir, "all_images.pt"),
        os.path.join(os.path.dirname(eval_dir), "all_images.pt"),
    )
    clipemb_path = resolve_existing_path(
        os.path.join(eval_dir, f"{args.model_name}_all_clipemb.pt"),
        os.path.join(eval_dir, f"{args.model_name}_all_clipvoxels.pt"),
    ) if (
        os.path.exists(os.path.join(eval_dir, f"{args.model_name}_all_clipemb.pt"))
        or os.path.exists(os.path.join(eval_dir, f"{args.model_name}_all_clipvoxels.pt"))
    ) else None

    all_recons = torch.load(recon_path, weights_only=False).float().clamp(0, 1)
    all_images = torch.load(gt_path, weights_only=False).float().clamp(0, 1)
    all_clipemb = torch.load(clipemb_path, weights_only=False).float() if clipemb_path is not None else None

    if all_recons.shape[-1] != all_images.shape[-1]:
        all_recons = transforms.Resize(all_images.shape[-2:])(all_recons)

    # --- PixCorr / SSIM ---
    pixcorr = safe_run(lambda: compute_pixcorr(all_images, all_recons, nan=True).item(), "PixCorr")

    def _ssim():
        img_gray = rgb2gray(transforms.Resize(425)(all_images).permute(0, 2, 3, 1).cpu().numpy())
        rec_gray = rgb2gray(transforms.Resize(425)(all_recons).permute(0, 2, 3, 1).cpu().numpy())
        vals = [
            ssim(rec_gray[i], img_gray[i], data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            for i in range(len(img_gray))
        ]
        return float(np.mean(vals))

    ssim_score = safe_run(_ssim, "SSIM")

    # --- 2-way identification metrics ---
    alex_pre = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    alex2 = safe_run(
        lambda: two_way_identification(
            all_recons,
            all_images,
            create_feature_extractor(
                alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
                return_nodes=["features.4", "features.11"],
            ).to(device).eval(),
            alex_pre,
            feature_key="features.4",
            device=device,
        ),
        "AlexNet(2)",
    )
    alex5 = safe_run(
        lambda: two_way_identification(
            all_recons,
            all_images,
            create_feature_extractor(
                alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
                return_nodes=["features.4", "features.11"],
            ).to(device).eval(),
            alex_pre,
            feature_key="features.11",
            device=device,
        ),
        "AlexNet(5)",
    )

    inc_pre = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    inception = safe_run(
        lambda: two_way_identification(
            all_recons,
            all_images,
            create_feature_extractor(
                inception_v3(weights=Inception_V3_Weights.DEFAULT),
                return_nodes=["avgpool"],
            ).to(device).eval(),
            inc_pre,
            feature_key="avgpool",
            device=device,
        ),
        "InceptionV3",
    )

    # CLIP metric using OpenCLIP bigG embedding 2-way identification
    def _clip_tw():
        embedder = build_clip_image_embedder(device)

        gt = embedder(all_images.to(device)).float().flatten(1).cpu().numpy()
        pred = embedder(all_recons.to(device)).float().flatten(1).cpu().numpy()
        r = np.corrcoef(gt, pred)
        r = r[: len(gt), len(gt) :]
        congruent = np.diag(r)
        success = r < congruent
        return float(np.mean(np.sum(success, axis=0)) / (len(gt) - 1))

    clip_metric = safe_run(_clip_tw, "CLIP")

    eff_pre = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _eff_dist():
        eff_model = create_feature_extractor(
            efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
            return_nodes=["avgpool"],
        ).to(device)
        eff_model.eval().requires_grad_(False)
        gt = eff_model(eff_pre(all_images.to(device)))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
        pred = eff_model(eff_pre(all_recons.to(device)))["avgpool"].reshape(len(all_recons), -1).cpu().numpy()
        return float(np.mean([distance.correlation(gt[i], pred[i]) for i in range(len(gt))]))

    effnet_b = safe_run(_eff_dist, "EffNet-B")

    def _swav_dist():
        swav = torch.hub.load("facebookresearch/swav:main", "resnet50")
        swav = create_feature_extractor(swav, return_nodes=["avgpool"]).to(device)
        swav.eval().requires_grad_(False)
        pre = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        gt = swav(pre(all_images.to(device)))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
        pred = swav(pre(all_recons.to(device)))["avgpool"].reshape(len(all_recons), -1).cpu().numpy()
        return float(np.mean([distance.correlation(gt[i], pred[i]) for i in range(len(gt))]))

    swav_metric = safe_run(_swav_dist, "SwAV")

    # --- Retrieval metrics ---
    if all_clipemb is not None:
        fwd_retrieval, bwd_retrieval = safe_run_obj(
            lambda: clip_fwd_bwd_retrieval(all_images, all_clipemb, device=device),
            "RetrievalTuple",
            (float("nan"), float("nan")),
        )
    else:
        print("[WARN] missing all_clipemb file; retrieval metrics set to NaN")
        fwd_retrieval, bwd_retrieval = float("nan"), float("nan")

    # --- Brain correlation metrics ---
    brain_metrics = safe_run_obj(
        lambda: brain_region_correlation(args, device=device),
        "BrainCorr",
        {
            "Brain Corr. nsd_general": float("nan"),
            "Brain Corr. V1": float("nan"),
            "Brain Corr. V2": float("nan"),
            "Brain Corr. V3": float("nan"),
            "Brain Corr. V4": float("nan"),
            "Brain Corr. higher_vis": float("nan"),
        },
    )

    metrics = {
        "PixCorr": pixcorr,
        "SSIM": ssim_score,
        "AlexNet(2)": alex2,
        "AlexNet(5)": alex5,
        "InceptionV3": inception,
        "CLIP": clip_metric,
        "EffNet-B": effnet_b,
        "SwAV": swav_metric,
        "FwdRetrieval": fwd_retrieval,
        "BwdRetrieval": bwd_retrieval,
        "Brain Corr. nsd_general": brain_metrics["Brain Corr. nsd_general"],
        "Brain Corr. V1": brain_metrics["Brain Corr. V1"],
        "Brain Corr. V2": brain_metrics["Brain Corr. V2"],
        "Brain Corr. V3": brain_metrics["Brain Corr. V3"],
        "Brain Corr. V4": brain_metrics["Brain Corr. V4"],
        "Brain Corr. higher_vis": brain_metrics["Brain Corr. higher_vis"],
    }

    df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    print(df.to_string(index=False))

    os.makedirs("tables", exist_ok=True)
    out_csv = os.path.join("tables", f"{args.model_name}_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved table: {out_csv}")


if __name__ == "__main__":
    main()

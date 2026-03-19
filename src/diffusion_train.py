import os
import sys
import argparse
import random
import h5py
import numpy as np
from tqdm import tqdm
import webdataset as wds

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

import utils
from models.model import FMRIModel
from models.diffusion import GaussianDiffusionTrainer
from models.mindeye_prior_snn import SNNPriorNetwork, BrainDiffusionPriorSNN

# SDXL unCLIP dependency path
sys.path.append("generative_models/")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

# tf32 is faster on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True


def load_model_ckpt(model, ckpt_path, key_candidates=("model", "model_state_dict")):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint
    for k in key_candidates:
        if k in checkpoint:
            state = checkpoint[k]
            break
    missing, unexpected = model.load_state_dict(state, strict=False)
    return len(missing), len(unexpected)


def save_ckpt(path, epoch, denoiser, trainer, optimizer, scheduler, best_eval):
    torch.save(
        {
            "epoch": epoch,
            "net_model": denoiser.state_dict(),
            "model": denoiser.state_dict(),
            "trainer": trainer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_eval": best_eval,
        },
        path,
    )


def get_valid_batch(dl_iter, max_tries=50):
    for _ in range(max_tries):
        behav, _, _, _ = next(dl_iter)
        image_idx = behav[:, 0, 0].cpu().long().numpy()
        image_unique, image_sorted_idx = np.unique(image_idx, return_index=True)
        if len(image_unique) != len(image_idx):
            continue
        voxel_idx = behav[:, 0, 5].cpu().long().numpy()
        voxel_sorted_idx = voxel_idx[image_sorted_idx]
        return image_unique, voxel_sorted_idx
    raise RuntimeError("Failed to fetch a duplicate-free batch from WebDataset.")


def build_fmri_model(args, voxel_dims, device):
    model = FMRIModel(
        voxel_dims=voxel_dims,
        hidden_dim=args.fmri_hidden_dim,
        fmri_feature_dim=args.fmri_feature_dim,
        fmri_feature_seq_len=args.fmri_feature_seq_len,
        timestep=args.fmri_timestep,
        ae_enc_depth=args.fmri_ae_enc_depth,
        ae_dec_depth=args.fmri_ae_dec_depth,
    ).to(device)
    return model


def extract_cond_tokens(features):
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


def make_scheduler(optimizer, scheduler_name, total_iters, max_lr):
    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    if scheduler_name == "cosine":
        eta_min = max_lr * 0.1
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_iters, 1), eta_min=eta_min)
    raise ValueError(f"Unknown scheduler={scheduler_name}")


@torch.no_grad()
def evaluate(
    fmri_model,
    diffusion_trainer,
    clip_img_embedder,
    voxels,
    images,
    test_dl,
    subj,
    subj_to_idx,
    device,
    max_batches=10,
):
    fmri_model.eval()
    diffusion_trainer.eval()
    clip_img_embedder.eval()

    losses = []
    subj_idx = subj_to_idx[subj]

    for i, (behav, _, _, _) in enumerate(test_dl):
        if i >= max_batches:
            break

        image_idx = behav[:, 0, 0].cpu().long().numpy()
        image_unique, image_sorted_idx = np.unique(image_idx, return_index=True)
        if len(image_unique) != len(image_idx):
            continue

        voxel_idx = behav[:, 0, 5].cpu().long().numpy()
        voxel_sorted_idx = voxel_idx[image_sorted_idx]

        voxel = voxels[f"subj0{subj}"][voxel_sorted_idx].float().unsqueeze(1).to(device, non_blocking=True)
        img_np = np.asarray(images[image_unique])
        img = torch.from_numpy(img_np).to(device=device, dtype=torch.float32)

        clip_target = clip_img_embedder(img).float()

        _, feats = fmri_model(voxel, subj_idx, returnFeatures=True)
        cond_tokens = extract_cond_tokens(feats)

        loss = diffusion_trainer(clip_target, cond=cond_tokens).mean()
        losses.append(loss.item())

    fmri_model.train()
    diffusion_trainer.train()
    return float(np.mean(losses)) if losses else float("inf")


def main():
    parser = argparse.ArgumentParser()

    # data & io
    parser.add_argument("--data_path", type=str, default="/data20TB/lzg/MindEyeV2")
    parser.add_argument("--model_name", type=str, default="diffusion_conditioned_clip")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # subject settings
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 5, 7])
    parser.add_argument("--finetune", action="store_true", help="if set, train only on args.subj")
    parser.add_argument("--multisubject_diffusion_ckpt", type=str, default=None, help="path to multisubject pretrained diffusion checkpoint for finetune")

    # pretrained fMRI model
    parser.add_argument("--fmri_ckpt", type=str, default="/data20TB/lzg/MindEyeV2/train_logs/fmri_encoder/final/best.pth", help="path to trained FMRIModel checkpoint")

    # new FMRIModel args
    parser.add_argument("--fmri_hidden_dim", type=int, default=4096)
    parser.add_argument("--fmri_feature_dim", type=int, default=1664)
    parser.add_argument("--fmri_feature_seq_len", type=int, default=256)
    parser.add_argument("--fmri_timestep", type=int, default=4)
    parser.add_argument("--fmri_ae_enc_depth", type=int, default=2)
    parser.add_argument("--fmri_ae_dec_depth", type=int, default=2)

    # prior diffusion model
    parser.add_argument("--clip_seq_dim", type=int, default=256)
    parser.add_argument("--clip_emb_dim", type=int, default=1664)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--prior_depth", type=int, default=6)
    parser.add_argument("--prior_heads", type=int, default=8)
    parser.add_argument("--prior_causal", action="store_true", help="enable causal attention in the prior")
    parser.add_argument("--prior_cond_drop_prob", type=float, default=0.20)
    parser.add_argument("--prior_learned_query_mode", type=str, default="pos_emb", choices=["none", "token", "pos_emb", "all_pos_emb"])

    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--overfit_single_batch", action="store_true", help="reuse one batch per subject for debugging")
    parser.add_argument("--resume", action="store_true", help="resume diffusion training from checkpoint")
    parser.add_argument("--resume_model", type=str, default=None, help="path to diffusion checkpoint for resume")

    args = parser.parse_args()

    utils.seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    outdir = os.path.abspath(f"/data20TB/lzg/MindEyeV2/train_logs/diffusion/{args.model_name}")
    os.makedirs(outdir, exist_ok=True)

    nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])

    if args.finetune:
        subj_list = np.array([args.subj])
        print(f"finetune mode: only subject {args.subj}")
    else:
        subj_list = np.arange(1, 9)
        print(f"multi-subject mode: {subj_list.tolist()}")

    subj_to_idx = {s: i for i, s in enumerate(subj_list)}

    # ------------------------------------------------
    # load wds loaders + voxel betas
    # ------------------------------------------------
    train_dl = {}
    voxels = {}
    voxel_dims = []

    batch_size_per_subj = max(1, args.batch_size // len(subj_list))
    num_samples_per_epoch = 750 * 40
    num_iterations_per_epoch = num_samples_per_epoch // (batch_size_per_subj * len(subj_list))
    print(
        f"batch_size_per_subject={batch_size_per_subj}, "
        f"num_iterations_per_epoch={num_iterations_per_epoch}"
    )

    for s in subj_list:
        train_url = f"{args.data_path}/wds/subj0{s}/train/{{0..{nsessions_allsubj[s - 1] - 1}}}.tar"
        print(f"train_url: {train_url}")

        data = (
            wds.WebDataset(train_url, resampled=True)
            .shuffle(750, initial=1500, rng=random.Random(42))
            .decode("torch")
            .rename(
                behav="behav.npy",
                past_behav="past_behav.npy",
                future_behav="future_behav.npy",
                olds_behav="olds_behav.npy",
            )
            .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
        )
        train_dl[f"subj0{s}"] = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size_per_subj,
            drop_last=False,
            pin_memory=True,
        )

        with h5py.File(f"{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5", "r") as f:
            betas = torch.tensor(f["betas"][:], dtype=torch.float32)
        voxels[f"subj0{s}"] = betas
        voxel_dims.append(betas.shape[-1])
        print(f"num_voxels for subj0{s}: {betas.shape[-1]}")

    # ------------------------------------------------
    # load test dl (eval only on args.subj)
    # ------------------------------------------------
    if args.subj not in subj_to_idx:
        raise ValueError(
            f"args.subj={args.subj} is not in active training subjects {subj_list.tolist()}"
        )

    test_url = f"{args.data_path}/wds/subj0{args.subj}/test/0.tar"
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
    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size_per_subj,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # ------------------------------------------------
    # load NSD images + CLIP image embedder
    # ------------------------------------------------
    f = h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r")
    images = f["images"]
    print("Loaded all 73k possible NSD images to cpu!", images.shape)

    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)
    clip_img_embedder.eval().requires_grad_(False)

    # ------------------------------------------------
    # load frozen FMRI model (feature extractor)
    # ------------------------------------------------
    fmri_model = build_fmri_model(args, voxel_dims, device)
    missing, unexpected = load_model_ckpt(fmri_model, args.fmri_ckpt, key_candidates=("model", "model_state_dict", "encoder"))
    print(f"Loaded FMRIModel ckpt: missing={missing}, unexpected={unexpected}")
    fmri_model.eval().requires_grad_(False)

    # ------------------------------------------------
    # SNN prior denoiser + diffusion trainer
    # ------------------------------------------------
    prior_net = SNNPriorNetwork(
        dim=args.clip_emb_dim,
        num_timesteps=args.T,
        depth=args.prior_depth,
        heads=args.prior_heads,
        dropout=args.dropout,
        num_time_embeds=1,
        num_tokens=args.clip_seq_dim,
        causal=args.prior_causal,
        learned_query_mode=args.prior_learned_query_mode,
    )
    denoiser = BrainDiffusionPriorSNN(prior_net, cond_drop_prob=args.prior_cond_drop_prob).to(device)

    if args.resume and args.resume_model is not None:
        print("resume mode enabled, skip --multisubject_diffusion_ckpt loading.")
    elif args.finetune and args.multisubject_diffusion_ckpt is not None:
        m, u = load_model_ckpt(
            denoiser,
            args.multisubject_diffusion_ckpt,
            key_candidates=("net_model", "model", "model_state_dict", "denoiser"),
        )
        print(f"Loaded multisubject diffusion ckpt: missing={m}, unexpected={u}")
    elif args.finetune:
        print("finetune enabled but --multisubject_diffusion_ckpt is None; training diffusion from scratch.")

    diffusion_trainer = GaussianDiffusionTrainer(
        model=denoiser,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
    ).to(device)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [p for n, p in denoiser.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in denoiser.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)
    scheduler = make_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        total_iters=int(np.floor(args.num_epochs * num_iterations_per_epoch)),
        max_lr=args.max_lr,
    )

    train_iters = {s: iter(train_dl[f"subj0{s}"]) for s in subj_list}
    fixed_batches = {} if args.overfit_single_batch else None

    start_epoch = 0
    best_eval = float("inf")
    if args.resume:
        if args.resume_model is None:
            raise ValueError("--resume requires --resume_model")
        ckpt = torch.load(os.path.abspath(args.resume_model), map_location="cpu")
        print(f"Loading Resume model from {args.resume_model}")
        state = ckpt.get("net_model", ckpt.get("model", ckpt.get("model_state_dict")))
        if state is None:
            raise KeyError("resume checkpoint missing net_model/model/model_state_dict")
        missing, unexpected = denoiser.load_state_dict(state, strict=False)
        print(f"resume denoiser loaded. missing={len(missing)}, unexpected={len(unexpected)}")

        if "trainer" in ckpt:
            diffusion_trainer.load_state_dict(ckpt["trainer"])
            print("Loaded trainer state.")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("Loaded optimizer state.")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("Loaded scheduler state.")

        start_epoch = int(ckpt.get("epoch", 0))
        best_eval = float(ckpt.get("best_eval", float("inf")))
        print(f"resume epoch={start_epoch}, best_eval={best_eval:.6f}")
    else:
        print("Training from scratch")

    if start_epoch >= args.num_epochs:
        print(
            f"checkpoint epoch ({start_epoch}) >= num_epochs ({args.num_epochs}), nothing to train. "
            "Increase --num_epochs to continue training."
        )
        return

    print(
        f"Start diffusion training from epoch {start_epoch + 1} to {args.num_epochs} "
        f"(lr={args.max_lr}, scheduler={args.scheduler}, causal={args.prior_causal}, cond_drop={args.prior_cond_drop_prob}, "
        f"overfit_single_batch={args.overfit_single_batch})"
    )

    for epoch in tqdm(range(start_epoch, args.num_epochs), ncols=150, desc="Epoch"):
        denoiser.train()
        running_loss = 0.0

        pbar = tqdm(range(num_iterations_per_epoch), ncols=150, desc="Iterations", leave=False)
        for it in pbar:
            optimizer.zero_grad()

            loss_sum = 0.0
            for s in subj_list:
                if fixed_batches is not None and s in fixed_batches:
                    image_unique, voxel_sorted_idx = fixed_batches[s]
                else:
                    while True:
                        try:
                            image_unique, voxel_sorted_idx = get_valid_batch(train_iters[s])
                            if fixed_batches is not None:
                                fixed_batches[s] = (image_unique, voxel_sorted_idx)
                            break
                        except StopIteration:
                            train_iters[s] = iter(train_dl[f"subj0{s}"])

                voxel = voxels[f"subj0{s}"][voxel_sorted_idx].float().unsqueeze(1).to(device, non_blocking=True)
                img_np = np.asarray(images[image_unique])
                img = torch.from_numpy(img_np).to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    _, feats = fmri_model(voxel, subj_to_idx[s], returnFeatures=True)
                    cond_tokens = extract_cond_tokens(feats)
                    clip_target = clip_img_embedder(img).float()

                if clip_target.shape[1] != args.clip_seq_dim or clip_target.shape[2] != args.clip_emb_dim:
                    raise RuntimeError(
                        f"Unexpected CLIP shape {tuple(clip_target.shape)}; "
                        f"expected [B,{args.clip_seq_dim},{args.clip_emb_dim}]"
                    )

                loss = diffusion_trainer(clip_target, cond=cond_tokens).mean()
                loss_sum += loss

            loss = loss_sum / len(subj_list)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(
                train_diff=f"{running_loss / (it + 1):.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_epoch_loss = running_loss / num_iterations_per_epoch
        print(
            f"epoch {epoch + 1}: train_diff_loss={train_epoch_loss:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if (epoch + 1) % args.eval_every == 0:
            eval_loss = evaluate(
                fmri_model=fmri_model,
                diffusion_trainer=diffusion_trainer,
                clip_img_embedder=clip_img_embedder,
                voxels=voxels,
                images=images,
                test_dl=test_dl,
                subj=args.subj,
                subj_to_idx=subj_to_idx,
                device=device,
                max_batches=args.eval_batches,
            )
            print(f"epoch {epoch + 1}: eval_diff_loss={eval_loss:.6f}")

            if eval_loss < best_eval:
                best_eval = eval_loss
                save_ckpt(
                    os.path.join(outdir, "best.pth"),
                    epoch + 1,
                    denoiser,
                    diffusion_trainer,
                    optimizer,
                    scheduler,
                    best_eval,
                )

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(
                os.path.join(outdir, f"epoch_{epoch + 1}.pth"),
                epoch + 1,
                denoiser,
                diffusion_trainer,
                optimizer,
                scheduler,
                best_eval,
            )

    save_ckpt(
        os.path.join(outdir, "last.pth"),
        args.num_epochs,
        denoiser,
        diffusion_trainer,
        optimizer,
        scheduler,
        best_eval,
    )
    print(f"training finished. best_eval_diff_loss={best_eval:.6f}")


if __name__ == "__main__":
    main()

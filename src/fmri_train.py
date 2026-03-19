import os
import argparse
import random
import glob
import re
import h5py
import numpy as np
from tqdm import tqdm
import webdataset as wds

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn

import utils
from models.model import FMRIModel

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/data20TB/lzg/MindEyeV2", help="Path to NSD data")
parser.add_argument("--device", type=int, default=0, help="which GPU to use for training")
parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 5, 7], help="subject for eval / finetune")

parser.add_argument("--model_name", type=str, default="fmri_transformer_snn_ae", help="run name for checkpoints/logs")
parser.add_argument("--seed", type=int, default=42, help="global random seed")

# model hyperparameters
parser.add_argument("--hidden_dim", type=int, default=4096, help="hidden dimension")
parser.add_argument("--fmri_feature_dim", type=int, default=1664, help="intermediate fmri feature dim")
parser.add_argument("--fmri_feature_seq_len", type=int, default=256, help="intermediate fmri feature seq len")
parser.add_argument("--timestep", type=int, default=4, help="SNN simulation timesteps")
parser.add_argument("--ae_enc_depth", type=int, default=4, help="shared AE encoder depth")
parser.add_argument("--ae_dec_depth", type=int, default=4, help="shared AE decoder depth")

# optimization
parser.add_argument("--batch_size", type=int, default=128, help="global batch size")
parser.add_argument("--max_lr", type=float, default=1e-4, help="max learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")

# training mode
parser.add_argument("--finetune", action="store_true", help="if set, train only on args.subj")
parser.add_argument("--finetune_ckpt", type=str, default=None, help="optional pretrained checkpoint path")

# io / eval
parser.add_argument("--eval_every", type=int, default=1, help="evaluate every N epochs")
parser.add_argument("--save_every", type=int, default=1, help="save checkpoint every N epochs")
parser.add_argument("--eval_batch_size", type=int, default=128, help="evaluation batch size to avoid OOM")
parser.add_argument("--load_ckpt", action="store_true", help="resume training from checkpoint in model_name dir")
parser.add_argument("--resume_ckpt", type=str, default=None, help="explicit checkpoint path for resume")

args = parser.parse_args()
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, test_dl, voxels, subj, subj_to_idx, criterion, device):
    model.eval()

    losses = []
    cond_norms = []
    subj_idx = subj_to_idx[subj]

    for behav, _, _, _ in test_dl:
        voxel_idx = behav[:, 0, 5].cpu().long().numpy()
        voxel = voxels[f"subj0{subj}"][voxel_idx].float().unsqueeze(1).to(device, non_blocking=True)

        recon, features = model(voxel, subj_idx, returnFeatures=True)
        loss = criterion(voxel, recon)

        losses.append(loss.item())
        cond_norms.append(features.norm(dim=-1).mean().item())

    model.train()
    return float(np.mean(losses)), float(np.mean(cond_norms))


def save_ckpt(path, epoch, model, optimizer, lr_scheduler, best_eval):
    encoder_state = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("recon_heads.")
    }
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "encoder": encoder_state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_eval": best_eval,
    }
    torch.save(ckpt, path)


def resolve_resume_ckpt(outdir, resume_ckpt):
    if resume_ckpt is not None:
        path = os.path.abspath(resume_ckpt)
        if not os.path.exists(path):
            raise FileNotFoundError(f"resume_ckpt not found: {path}")
        return path

    last_path = os.path.join(outdir, "last.pth")
    if os.path.exists(last_path):
        return last_path

    epoch_paths = glob.glob(os.path.join(outdir, "epoch_*.pth"))
    if not epoch_paths:
        raise FileNotFoundError(
            f"No checkpoint found in {outdir}. Expected last.pth or epoch_*.pth, "
            "or pass --resume_ckpt explicitly."
        )

    def extract_epoch(path):
        m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    epoch_paths.sort(key=extract_epoch)
    return epoch_paths[-1]


def main():
    outdir = os.path.abspath(f"/data20TB/lzg/MindEyeV2/train_logs/fmri_encoder/{args.model_name}")
    os.makedirs(outdir, exist_ok=True)

    utils.seed_everything(args.seed)

    nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
    if args.finetune:
        subj_list = np.array([args.subj])
        print(f"finetune mode: only train subj{args.subj:02d}")
    else:
        subj_list = np.arange(1, 9)
        print(f"multi-subject mode: train subjects {subj_list.tolist()}")

    subj_to_idx = {s: i for i, s in enumerate(subj_list)}

    # divide global batch by number of active subjects
    batch_size = max(1, args.batch_size // len(subj_list))
    num_samples_per_epoch = 750 * 40
    num_iterations_per_epoch = num_samples_per_epoch // (batch_size * len(subj_list))
    print(
        f"batch_size_per_subject={batch_size}, num_iterations_per_epoch={num_iterations_per_epoch}, "
        f"num_samples_per_epoch={num_samples_per_epoch}"
    )

    # ------------------------------------------------
    # load training data and voxels
    # ------------------------------------------------
    train_dl = {}
    voxels = {}
    num_voxels_list = []

    for s in subj_list:
        train_url = f"{args.data_path}/wds/subj0{s}/train/{{0..{nsessions_allsubj[s - 1] - 1}}}.tar"
        print(f"train_url: {train_url}")

        train_data = (
            wds.WebDataset(train_url, resampled=True, shardshuffle=False)
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
            train_data,
            batch_size=batch_size,
            drop_last=False,
            pin_memory=True,
        )

        with h5py.File(f"{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5", "r") as f:
            betas = torch.tensor(f["betas"][:], dtype=torch.float32)

        voxels[f"subj0{s}"] = betas
        num_voxels_list.append(betas.shape[-1])
        print(f"num_voxels for subj0{s}: {betas.shape[-1]}")

    train_dls = [train_dl[f"subj0{s}"] for s in subj_list]
    print("Loaded train loaders and voxel betas.")

    # ------------------------------------------------
    # load test data (always evaluate on args.subj)
    # ------------------------------------------------
    subj = args.subj
    if subj not in subj_to_idx:
        raise ValueError(
            f"args.subj={subj} is not in current training subjects {subj_list.tolist()}. "
            f"Set --finetune or pick a subject within training set."
        )

    test_url = f"{args.data_path}/wds/subj0{subj}/test/0.tar"
    print(f"test_url: {test_url}")

    test_data = (
        wds.WebDataset(test_url, resampled=False, shardshuffle=False)
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
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    print(f"eval_batch_size={args.eval_batch_size}")

    # ------------------------------------------------
    # build model
    # ------------------------------------------------
    fmri_model = FMRIModel(
        voxel_dims=num_voxels_list,
        hidden_dim=args.hidden_dim,
        fmri_feature_dim=args.fmri_feature_dim,
        fmri_feature_seq_len=args.fmri_feature_seq_len,
        timestep=args.timestep,
        ae_enc_depth=args.ae_enc_depth,
        ae_dec_depth=args.ae_dec_depth,
    ).to(device)

    if args.finetune_ckpt is not None:
        print(f"loading finetune checkpoint: {args.finetune_ckpt}")
        ckpt = torch.load(args.finetune_ckpt, map_location="cpu")
        state = ckpt["encoder"] if "encoder" in ckpt else (ckpt["model"] if "model" in ckpt else ckpt)
        missing, unexpected = fmri_model.load_state_dict(state, strict=False)
        print(f"ckpt loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    fmri_model.train()

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in fmri_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in fmri_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.max_lr)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(args.num_epochs * num_iterations_per_epoch)),
        last_epoch=-1,
    )

    mse = nn.MSELoss()

    start_epoch = 0
    best_eval = float("inf")
    if args.load_ckpt or args.resume_ckpt is not None:
        resume_path = resolve_resume_ckpt(outdir, args.resume_ckpt)
        print(f"resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")

        state = ckpt["model"] if "model" in ckpt else (ckpt["encoder"] if "encoder" in ckpt else ckpt)
        missing, unexpected = fmri_model.load_state_dict(state, strict=False)
        print(f"model resumed. missing={len(missing)}, unexpected={len(unexpected)}")

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("optimizer state resumed.")
        else:
            print("optimizer state not found in checkpoint, using fresh optimizer.")

        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            print("lr scheduler state resumed.")
        else:
            print("lr scheduler state not found in checkpoint, using fresh scheduler.")

        start_epoch = int(ckpt.get("epoch", 0))
        best_eval = float(ckpt.get("best_eval", float("inf")))
        print(f"resume epoch={start_epoch}, best_eval={best_eval:.6f}")

    if start_epoch >= args.num_epochs:
        print(
            f"checkpoint epoch ({start_epoch}) >= num_epochs ({args.num_epochs}), nothing to train. "
            "Increase --num_epochs to continue training."
        )
        return

    print(f"{args.model_name} start training from epoch {start_epoch + 1} to {args.num_epochs}")

    for epoch in tqdm(range(start_epoch, args.num_epochs), ncols=150, desc="Epoch"):
        voxel_iters = {}
        available_iters = []

        # preload batches for each subject
        for s_idx, dl in enumerate(train_dls):
            s = subj_list[s_idx]
            it_count = -1

            for behav0, _, _, _ in dl:
                image_idx = behav0[:, 0, 0].cpu().long().numpy()
                image_unique, image_sorted_idx = np.unique(image_idx, return_index=True)

                # hdf5 indexing with duplicates can fail
                if len(image_unique) != len(image_idx):
                    continue

                it_count += 1
                voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
                voxel_sorted_idx = voxel_idx[image_sorted_idx]

                voxel0 = voxels[f"subj0{s}"][voxel_sorted_idx].float().unsqueeze(1)
                voxel_iters[f"subj0{s}_iter{it_count}"] = voxel0

                if it_count >= num_iterations_per_epoch - 1:
                    break

            available_iters.append(it_count + 1)

        cur_iters = min(available_iters) if available_iters else 0
        if cur_iters <= 0:
            raise RuntimeError("No valid training batches were prepared. Check dataset and indexing.")

        running_loss = 0.0
        pbar = tqdm(range(cur_iters), ncols=150, desc="Iterations", leave=False)
        for it in pbar:
            optimizer.zero_grad()

            voxel_list = [voxel_iters[f"subj0{s}_iter{it}"].detach().to(device) for s in subj_list]

            loss_voxel = 0.0
            for s_idx in range(len(subj_list)):
                recon, _ = fmri_model(voxel_list[s_idx], s_idx, returnFeatures=True)
                loss_voxel += mse(voxel_list[s_idx], recon)

            loss_voxel = loss_voxel / len(subj_list)
            loss_voxel.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss_voxel.item()
            pbar.set_postfix(train_mse=f"{running_loss / (it + 1):.6f}")

        train_epoch_loss = running_loss / cur_iters
        print(f"epoch {epoch + 1}: train_mse={train_epoch_loss:.6f}")

        # eval
        if (epoch + 1) % args.eval_every == 0:
            eval_mse, cond_norm = evaluate(
                fmri_model,
                test_dl,
                voxels,
                subj,
                subj_to_idx,
                mse,
                device,
            )
            print(f"epoch {epoch + 1}: eval_mse={eval_mse:.6f}, cond_norm={cond_norm:.6f}")

            if eval_mse < best_eval:
                best_eval = eval_mse
                save_ckpt(
                    os.path.join(outdir, "best.pth"),
                    epoch + 1,
                    fmri_model,
                    optimizer,
                    lr_scheduler,
                    best_eval,
                )

        # periodic save
        if (epoch + 1) % args.save_every == 0:
            save_ckpt(
                os.path.join(outdir, f"epoch_{epoch + 1}.pth"),
                epoch + 1,
                fmri_model,
                optimizer,
                lr_scheduler,
                best_eval,
            )

    save_ckpt(
        os.path.join(outdir, "last.pth"),
        args.num_epochs,
        fmri_model,
        optimizer,
        lr_scheduler,
        best_eval,
    )
    print(f"training finished. best_eval_mse={best_eval:.6f}")


if __name__ == "__main__":
    main()

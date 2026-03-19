import os
import argparse
import random

import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize random original vs reconstructed image pairs")
    parser.add_argument("--model_name", type=str, default="fmri_diffusion_recon")
    parser.add_argument("--eval_dir", type=str, default=None, help="Directory containing recon outputs")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    random.seed(args.seed)

    eval_dir = args.eval_dir or os.path.abspath(f"evals/{args.model_name}")
    recon_path = os.path.join(eval_dir, f"{args.model_name}_all_recons.pt")
    gt_path = os.path.join(eval_dir, "all_images.pt")

    if not os.path.exists(recon_path):
        raise FileNotFoundError(f"Missing recon file: {recon_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing gt file: {gt_path}")

    recons = torch.load(recon_path, weights_only=False).float().clamp(0, 1)
    images = torch.load(gt_path, weights_only=False).float().clamp(0, 1)

    if len(recons) != len(images):
        raise ValueError(f"Count mismatch: recons={len(recons)} images={len(images)}")

    if recons.shape[-2:] != images.shape[-2:]:
        recons = transforms.Resize(images.shape[-2:])(recons)

    n_total = len(images)
    n = min(args.num_samples, n_total)
    idx = random.sample(range(n_total), n)

    if args.resize > 0:
        resize_tf = transforms.Resize((args.resize, args.resize))
        images_vis = resize_tf(images[idx])
        recons_vis = resize_tf(recons[idx])
    else:
        images_vis = images[idx]
        recons_vis = recons[idx]

    # 20 rows x 2 cols: [Original | Recon]
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, max(2 * n, 10)))
    if n == 1:
        axes = [axes]

    for row in range(n):
        gt = images_vis[row].permute(1, 2, 0).cpu().numpy()
        rc = recons_vis[row].permute(1, 2, 0).cpu().numpy()

        axes[row][0].imshow(gt)
        axes[row][0].set_title(f"Original #{idx[row]}", fontsize=9)
        axes[row][0].axis("off")

        axes[row][1].imshow(rc)
        axes[row][1].set_title("Recon", fontsize=9)
        axes[row][1].axis("off")

    plt.tight_layout()

    output_path = args.output or os.path.join(eval_dir, f"{args.model_name}_random20_compare.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()

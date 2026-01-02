import argparse
import os

"""
args_names = [
    "model_name",           # 模型名称，用于保存检查点和wandb日志
    "data_path",            # NSD数据存储/下载路径
    "cache_dir",            # huggingface下载文件的存储路径
    "subj",                 # 验证的主题ID (1-8)
    "multisubject_ckpt",    # 预训练的多主题模型路径（用于微调）
    "num_sessions",         # 包含的训练会话数量
    "use_prior",            # 是否训练扩散先验
    "batch_size",           # 批次大小
    "wandb_log",            # 是否启用wandb日志
    "wandb_project",        # wandb项目名称
    "mixup_pct",            # 从BiMixCo切换到SoftCLIP的训练比例
    "blurry_recon",         # 是否输出模糊重建
    "blur_scale",           # 模糊重建损失权重
    "clip_scale",           # 对比损失权重
    "prior_scale",          # 扩散先验损失权重
    "use_image_aug",        # 是否使用图像增强
    "num_epochs",           # 训练轮数
    "multi_subject",        # 是否多主题训练
    "new_test",             # 是否使用新的测试集
    "n_blocks",             # 块数（网络结构参数）
    "hidden_dim",           # 隐藏层维度
    "lr_scheduler_type",    # 学习率调度器类型（cycle/linear）
    "ckpt_saving",          # 是否保存检查点
    "ckpt_interval",        # 保存检查点/重建的间隔轮数
    "seed",                 # 随机种子
    "max_lr"                # 最大学习率
]
"""


def get_train_args():

    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name",
        type=str,
        default="testing",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data20TB/ZZZ/MindEyeV2",
        help="Path to where NSD data is stored / where to download it to",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data20TB/ZZZ/MindEyeV2",
        help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
    )
    parser.add_argument(
        "--subj",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--multisubject_ckpt",
        type=str,
        default=None,
        help="Path to pre-trained multisubject model to finetune a single subject from. multisubject must be False.",
    )
    parser.add_argument(
        "--num_sessions",
        type=int,
        default=1,
        help="Number of training sessions to include",
    )
    parser.add_argument(
        "--use_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
    )
    parser.add_argument(
        "--wandb_log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to log to wandb",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stability",
        help="wandb project name",
    )
    parser.add_argument(
        "--mixup_pct",
        type=float,
        default=0.33,
        help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--blurry_recon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--blur_scale",
        type=float,
        default=0.5,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--clip_scale",
        type=float,
        default=1.0,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--prior_scale",
        type=float,
        default=30,
        help="multiply diffusion prior loss by this",
    )
    parser.add_argument(
        "--use_image_aug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to use image augmentation",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=150,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--multi_subject",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--new_test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cycle",
        choices=["cycle", "linear"],
    )
    parser.add_argument(
        "--ckpt_saving",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=5,
        help="save backup ckpt and reconstruct every x epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--T",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--neuron_type",
        type=str,
        default="LIF",
    )
    parser.add_argument(
        "--detach_reset",
        type=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser(description="Model Testing Configuration")

    # MindAligner arguments
    parser.add_argument("--n_subj", type=int, default=1, choices=[1, 2, 5, 7])
    parser.add_argument("--k_subj", type=int, default=2, choices=[1, 2, 5, 7])
    parser.add_argument("--bfa_latent", type=int, default=4096)

    parser.add_argument(
        "--plotting", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--new_test", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)

    # decoding model arguments
    parser.add_argument(
        "--decoding_model_path", type=str, default=".../your_data_path/decoding_model"
    )
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--n_blocks", type=int, default=4)

    # data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=".../your_data_path",
        help="path to where NSD data is stored / where to download it to",
    )

    # loss arguments
    parser.add_argument(
        "--clip_scale",
        type=float,
        default=1.0,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--blurry_recon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--blur_scale",
        type=float,
        default=0.5,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--use_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--prior_scale",
        type=float,
        default=30,
        help="multiply diffusion prior loss by this",
    )

    return parser.parse_args()

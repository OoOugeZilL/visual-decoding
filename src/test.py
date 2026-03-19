import torch
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from tqdm import tqdm
import os
from torchvision import transforms
import torch.nn.functional as F
import sys

sys.path.append("generative_models/")
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
from omegaconf import OmegaConf
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims

def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04, device='cuda:3'):
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img

        # clip_img_tokenized = clip_img_embedder(image) 
        # tokens = clip_img_tokenized
        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples

def load_coco_images_from_hdf5(hdf5_path, num_images=10):
    """从HDF5文件加载COCO图片"""
    np.random.seed(42)
    images = []
    with h5py.File(hdf5_path, 'r') as f:
        # 假设数据集名为'images'，需要根据实际结构调整
        dataset = f['images']  # 可能需要改成实际的数据集名
        total_images = len(dataset)
        if num_images < total_images:
            indices = np.random.choice(total_images, num_images, replace=False)
        for i in indices:
            img = torch.from_numpy(dataset[i]).float()
            # 确保shape是 [3, 224, 224]
            if img.shape[-1] == 3:  # 如果是 HWC
                img = img.permute(2, 0, 1)
            images.append(img)
    return torch.stack(images) if images else None

def add_gaussian_noise(embedding, noise_level):
    """添加高斯噪声"""
    noise = torch.randn_like(embedding) * noise_level
    return embedding + noise

def gaussian_blur_embedding(embedding, kernel_size=3, sigma=1.0):
    """对embedding做高斯模糊（模拟丢弃高频信息）"""
    # embedding shape: [1, 256, 1664] -> 看作256个token，每个1664维
    # 我们在token维度上做模糊，相当于让相邻token的信息混合
    device = embedding.device
    dtype = embedding.dtype
    blurred = embedding.clone()
    
    # 创建1D高斯核
    kernel_1d = torch.arange(kernel_size).float() - kernel_size//2
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma)**2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # 扩展到2D卷积核
    kernel_2d = torch.outer(kernel_1d, kernel_1d)  # 形状: [kernel_size, kernel_size]
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).to(device=device, dtype=dtype)
    
    # 将embedding reshape为 [1, 1, 256, 1664] 进行2D卷积
    emb_2d = embedding.view(1, 1, 256, 1664)
    
    # 添加padding保持尺寸不变
    padding = kernel_size // 2
    blurred_2d = F.conv2d(emb_2d, kernel, padding=padding)
    
    return blurred_2d.view(1, 256, 1664)

def test_noise_sensitivity(images, clip_img_embedder, diffusion_engine, vector_suffix, 
                          noise_levels=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
                          blur_sigmas=[0, 0.5, 1.0, 2.0, 3.0, 5.0],
                          num_samples_per_image=1, device='cuda'):
    """测试不同扰动下的还原效果"""
    
    results = {
        'gaussian_noise': {'ssim_scores': [], 'images': []},
        'gaussian_blur': {'ssim_scores': [], 'images': []}
    }
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])
    
    for idx, img in enumerate(tqdm(images, desc="Processing images")):
        img = img.unsqueeze(0).to(device)  # [1, 3, 224, 224]
        
        # 1. 提取原始CLIP embedding
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            original_emb = clip_img_embedder(img)  # [1, 256, 1664]
        
        # 2. 对每张图片测试不同强度的噪声
        img_results_noise = {'levels': [], 'ssims': [], 'reconstructions': []}
        img_results_blur = {'sigmas': [], 'ssims': [], 'reconstructions': []}
        
        # 测试高斯噪声
        for noise_level in noise_levels:
            # 添加扰动
            perturbed_emb = add_gaussian_noise(original_emb.clone(), noise_level)
            perturbed_emb = perturbed_emb.to(device)
            
            # 用UNCLIP还原
            with torch.no_grad():
                recon_img = unclip_recon(perturbed_emb.half(), diffusion_engine, 
                                        vector_suffix, num_samples=num_samples_per_image, device=device)
            
            # 计算SSIM (需要转换为numpy并确保在0-1范围内)
            img_gray = rgb2gray(preprocess(img).cpu().squeeze(0).permute(1, 2, 0))
            recon_gray = rgb2gray(preprocess(recon_img).cpu().squeeze(0).permute(1, 2, 0))
            
            # 多通道SSIM
            ssim_score = ssim(img_gray, recon_gray, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            
            img_results_noise['levels'].append(noise_level)
            img_results_noise['ssims'].append(ssim_score)
            img_results_noise['reconstructions'].append(recon_img.cpu())
        
        # 测试高斯模糊
        for sigma in blur_sigmas:
            if sigma == 0:
                perturbed_emb = original_emb.clone()
            else:
                perturbed_emb = gaussian_blur_embedding(original_emb.clone(), kernel_size=5, sigma=sigma)
            perturbed_emb = perturbed_emb.to(device)
            
            with torch.no_grad():
                recon_img = unclip_recon(perturbed_emb.half(), diffusion_engine, 
                                        vector_suffix, num_samples=num_samples_per_image, device=device)
            
            img_gray = rgb2gray(preprocess(img).cpu().squeeze(0).permute(1, 2, 0))
            recon_gray = rgb2gray(preprocess(recon_img).cpu().squeeze(0).permute(1, 2, 0))
            ssim_score = ssim(img_gray, recon_gray, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            
            img_results_blur['sigmas'].append(sigma)
            img_results_blur['ssims'].append(ssim_score)
            img_results_blur['reconstructions'].append(recon_img.cpu())
        
        # 保存结果
        results['gaussian_noise']['ssim_scores'].append(img_results_noise['ssims'])
        results['gaussian_noise']['images'].append({
            'original': img.cpu(),
            'reconstructions': img_results_noise['reconstructions'],
            'levels': img_results_noise['levels']
        })
        
        results['gaussian_blur']['ssim_scores'].append(img_results_blur['ssims'])
        results['gaussian_blur']['images'].append({
            'original': img.cpu(),
            'reconstructions': img_results_blur['reconstructions'],
            'sigmas': img_results_blur['sigmas']
        })
    
    return results

def visualize_results(results, save_dir='sensitivity_results'):
    """可视化测试结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 绘制SSIM随扰动强度的变化曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 高斯噪声结果
    noise_ssims = np.array(results['gaussian_noise']['ssim_scores'])
    noise_levels = results['gaussian_noise']['images'][0]['levels']
    
    mean_ssim_noise = np.mean(noise_ssims, axis=0)
    std_ssim_noise = np.std(noise_ssims, axis=0)
    
    axes[0].errorbar(noise_levels, mean_ssim_noise, yerr=std_ssim_noise, 
                     marker='o', capsize=5, color='blue')
    axes[0].set_xlabel('Gaussian Noise Level')
    axes[0].set_ylabel('SSIM')
    axes[0].set_title('SSIM vs Gaussian Noise Level')
    axes[0].grid(True, alpha=0.3)
    
    # 高斯模糊结果
    blur_ssims = np.array(results['gaussian_blur']['ssim_scores'])
    blur_sigmas = results['gaussian_blur']['images'][0]['sigmas']
    
    mean_ssim_blur = np.mean(blur_ssims, axis=0)
    std_ssim_blur = np.std(blur_ssims, axis=0)
    
    axes[1].errorbar(blur_sigmas, mean_ssim_blur, yerr=std_ssim_blur, 
                     marker='s', capsize=5, color='red')
    axes[1].set_xlabel('Gaussian Blur Sigma')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs Gaussian Blur Strength')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ssim_curves.png'), dpi=150)
    plt.show()
    
    # 2. 展示一些示例图片
    num_examples = min(5, len(results['gaussian_noise']['images']))
    fig, axes = plt.subplots(num_examples, len(noise_levels) + len(blur_sigmas) + 1, 
                             figsize=(20, 3*num_examples))
    
    for i in range(num_examples):
        # 原始图片
        orig_img = results['gaussian_noise']['images'][i]['original'][0]
        axes[i, 0].imshow(orig_img.permute(1,2,0).numpy())
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # 高斯噪声还原结果
        for j, noise_level in enumerate(noise_levels):
            recon_img = results['gaussian_noise']['images'][i]['reconstructions'][j][0]
            axes[i, j+1].imshow(recon_img.permute(1,2,0).numpy())
            axes[i, j+1].set_title(f'Noise: {noise_level}')
            axes[i, j+1].axis('off')
        
        # 高斯模糊还原结果
        for j, sigma in enumerate(blur_sigmas):
            recon_img = results['gaussian_blur']['images'][i]['reconstructions'][j][0]
            axes[i, j+len(noise_levels)+1].imshow(recon_img.permute(1,2,0).numpy())
            axes[i, j+len(noise_levels)+1].set_title(f'Blur σ: {sigma}')
            axes[i, j+len(noise_levels)+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction_examples.png'), dpi=150)
    plt.show()

def main():
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载数据
    print("Loading COCO images...")
    images = load_coco_images_from_hdf5('/data20TB/lzg/MindEyeV2/coco_images_224_float16.hdf5', num_images=1000)
    if images is None:
        print("Failed to load images")
        return
    
    # 2. 初始化CLIP embedder (假设已经在外部初始化好)
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)

    # prep unCLIP
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
    offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38
    sampler_config['params']['device'] = device

    diffusion_engine = DiffusionEngine(network_config=network_config,
                            denoiser_config=denoiser_config,
                            first_stage_config=first_stage_config,
                            conditioner_config=conditioner_config,
                            sampler_config=sampler_config,
                            scale_factor=scale_factor,
                            disable_first_stage_autocast=disable_first_stage_autocast)
    # set to inference
    diffusion_engine.to(device)
    diffusion_engine.eval().requires_grad_(False)

    ckpt_path = '/data20TB/lzg/MindEyeV2/unclip6_epoch0_step110000.ckpt'
    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion_engine.load_state_dict(ckpt['state_dict'])

    batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
          "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
          "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    print("vector_suffix", vector_suffix.shape)
    
    # 这里需要你传入实际初始化的模型
    # 测试代码时可以用mock数据
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    
    # 3. 运行敏感性测试
    results = test_noise_sensitivity(images, clip_img_embedder, diffusion_engine, 
                                      vector_suffix, device=device)
    
    # 4. 可视化结果
    visualize_results(results)
    
    print("Ready to run sensitivity test with your initialized models!")

if __name__ == "__main__":
    main()
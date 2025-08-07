import os
from PIL import Image
import warnings
import csv
import trimesh
import numpy as np
import zipfile
import io
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from lpips import LPIPS
from omegaconf import OmegaConf
from tsr.system import TSR
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class PNGDataset(Dataset):
    def __init__(self, folder_path, gt_folder=None, trellis_folder=None, image_size=512):
        self.folder_path = folder_path
        self.gt_folder = gt_folder
        self.trellis_folder = trellis_folder
        self.image_size = image_size

        # 收集 PNG 檔案及其對應的 STL ZIP 和 GLB
        self.image_files = []
        self.gt_mapping = {}
        self.trellis_mapping = {}
        self.subfolders = set()

        # 掃描 PNG 資料夾
        for stl_folder in os.listdir(folder_path):
            stl_dir = os.path.join(folder_path, stl_folder)
            if os.path.isdir(stl_dir):
                for subfolder in os.listdir(stl_dir):
                    subfolder_dir = os.path.join(stl_dir, subfolder)
                    if os.path.isdir(subfolder_dir):
                        self.subfolders.add(subfolder)
                        png_files = [f for f in os.listdir(subfolder_dir) if f.endswith('.png')]
                        for png in png_files:
                            self.image_files.append((stl_folder, subfolder, png))

                        # STL ZIP
                        if gt_folder:
                            zip_name = f"{subfolder.replace('stl_', '')}.zip"
                            zip_path = os.path.join(gt_folder, zip_name)
                            if os.path.exists(zip_path):
                                self.gt_mapping[subfolder] = zip_path

        # 掃描 GLB 檔案
        if trellis_folder:
            glb_files = [f for f in os.listdir(trellis_folder) if f.endswith('.glb')]
            for subfolder in self.subfolders:
                # 嘗試匹配 GLB（例如 stl_27-3.glb 或 stl_27-3-1.glb）
                for glb in glb_files:
                    glb_base = os.path.splitext(glb)[0]
                    # 精確匹配或前綴匹配（例如 stl_49 匹配 stl_49-1.glb）
                    if glb_base == subfolder or glb_base.startswith(subfolder):
                        glb_path = os.path.join(trellis_folder, glb)
                        self.trellis_mapping[subfolder] = glb_path
                        break
                # 檢查子資料夾結構
                if subfolder not in self.trellis_mapping:
                    glb_path = os.path.join(trellis_folder, subfolder, f"{subfolder}.glb")
                    if os.path.exists(glb_path):
                        self.trellis_mapping[subfolder] = glb_path

        # 除錯資訊
        unmatched_subfolders = self.subfolders - set(self.trellis_mapping.keys())
        unmatched_glbs = [glb for glb in glb_files if not any(glb_base == s or glb_base.startswith(s) for s in self.subfolders for glb_base in [os.path.splitext(glb)[0]])]
        print(f"Found {len(self.image_files)} PNG files")
        print(f"Found {len(self.trellis_mapping)} GLB files")
        if unmatched_subfolders:
            print(f"Unmatched subfolders (no GLB): {sorted(unmatched_subfolders)}")
        if unmatched_glbs:
            print(f"Unmatched GLBs (no subfolder): {sorted(unmatched_glbs)}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if not self.image_files:
            raise ValueError(f"No valid PNG files found in {folder_path}")
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        stl_folder, subfolder, png_file = self.image_files[idx]
        img_path = os.path.join(self.folder_path, stl_folder, subfolder, png_file)
        default_output = {
            "image": torch.zeros((3, self.image_size, self.image_size)),
            "gt_path": None,
            "trellis_path": None,
            "valid": False  # 標記是否有效
        }

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            if image.sum() == 0:
                print(f"Invalid image (zero sum) at index {idx}: {img_path}")
                return default_output
            default_output["image"] = image
            default_output["valid"] = True
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return default_output

        gt_path = self.gt_mapping.get(subfolder) if self.gt_folder else None
        trellis_path = self.trellis_mapping.get(subfolder) if self.trellis_folder else None
        default_output["gt_path"] = gt_path
        default_output["trellis_path"] = trellis_path
        return default_output

# warnings.filterwarnings("ignore", category=UserWarning, message="Using torch.cross without specifying the dim arg")

def compute_loss_and_metrics(model, scene_codes, input_images, gt_path, trellis_path, device, lpips_loss_fn, batch_idx, epoch):
    mse_loss = torch.tensor(0.0, device=device)
    lpips_loss = torch.tensor(0.0, device=device)
    if input_images.sum() != 0:
        try:
            rendered_images = model.render(scene_codes, n_views=1, height=256, width=256, return_type="pt")
            rendered_image = rendered_images[0][0]
            rendered_image = rendered_image.permute(2, 0, 1)[None, ...].to(device)
            input_images_resized = F.interpolate(input_images, size=(256, 256), mode='bilinear', align_corners=False).to(device)
            mse_loss = nn.MSELoss()(rendered_image, input_images_resized)
            lpips_loss = lpips_loss_fn(rendered_image, input_images_resized).mean()
        except Exception as e:
            print(f"Error computing image loss for batch {batch_idx + 1}: {e}")
            logging.error(f"Batch {batch_idx + 1}: Error computing image loss: {e}")
            mse_loss = torch.tensor(0.0, device=device)
            lpips_loss = torch.tensor(0.0, device=device)

    cd_trellis = None
    cd_stl = None
    try:
        meshes = model.extract_mesh(scene_codes, False, resolution=192)
        gen_mesh = meshes[0]
        gen_points_np = trimesh.sample.sample_surface(gen_mesh, 10000)[0]
        # 正規化：移動到原點，縮放到單位範圍
        gen_points_np -= np.mean(gen_points_np, axis=0)  # 減去質心
        scale = np.max(np.abs(gen_points_np))
        if scale > 1e-8:  # 避免除以零
            gen_points_np /= scale
        gen_points = torch.tensor(gen_points_np, dtype=torch.float32, device=device)
    except Exception as e:
        print(f"Error extracting mesh for batch {batch_idx + 1}: {e}")
        logging.error(f"Batch {batch_idx + 1}: Error extracting mesh: {e}")
        return None, mse_loss, lpips_loss, cd_trellis, cd_stl

    if trellis_path and os.path.exists(trellis_path):
        try:
            loaded = trimesh.load(trellis_path, file_type='glb')
            if isinstance(loaded, trimesh.Scene):
                meshes = list(loaded.geometry.values())
                if not meshes:
                    print(f"Error: Empty Scene in GLB {trellis_path} (no valid meshes)")
                    logging.error(f"Batch {batch_idx + 1}: Empty Scene in GLB {trellis_path}")
                    return None, mse_loss, lpips_loss, cd_trellis, cd_stl
                trellis_mesh = meshes[0]
                print(f"Loaded Scene from {trellis_path} with {len(meshes)} meshes, using first mesh")
                logging.info(f"Batch {batch_idx + 1}: Loaded Scene from {trellis_path} with {len(meshes)} meshes")
            else:
                trellis_mesh = loaded
                print(f"Loaded Trimesh from {trellis_path}")
                logging.info(f"Batch {batch_idx + 1}: Loaded Trimesh from {trellis_path}")

            # 檢查 Mesh 有效性
            if not hasattr(trellis_mesh, 'vertices') or len(trellis_mesh.vertices) == 0:
                print(f"Error: Invalid Mesh in GLB {trellis_path} (no vertices)")
                logging.error(f"Batch {batch_idx + 1}: Invalid Mesh in GLB {trellis_path} (no vertices)")
                return None, mse_loss, lpips_loss, cd_trellis, cd_stl
            if not hasattr(trellis_mesh, 'faces') or len(trellis_mesh.faces) == 0:
                print(f"Error: Invalid Mesh in GLB {trellis_path} (no faces)")
                logging.error(f"Batch {batch_idx + 1}: Invalid Mesh in GLB {trellis_path} (no faces)")
                return None, mse_loss, lpips_loss, cd_trellis, cd_stl
            print(f"Mesh stats: {len(trellis_mesh.vertices)} vertices, {len(trellis_mesh.faces)} faces")
            logging.info(f"Batch {batch_idx + 1}: Mesh stats: {len(trellis_mesh.vertices)} vertices, {len(trellis_mesh.faces)} faces")

            trellis_points_np = trimesh.sample.sample_surface(trellis_mesh, 10000)[0]
            # 正規化：移動到原點，縮放到單位範圍
            trellis_points_np -= np.mean(trellis_points_np, axis=0)  # 減去質心
            scale = np.max(np.abs(trellis_points_np))
            if scale > 1e-8:  # 避免除以零
                trellis_points_np /= scale
            trellis_points = torch.tensor(trellis_points_np, dtype=torch.float32, device=device)
            dist1 = torch.cdist(gen_points.unsqueeze(0), trellis_points.unsqueeze(0)).min(dim=2)[0]
            dist2 = torch.cdist(trellis_points.unsqueeze(0), gen_points.unsqueeze(0)).min(dim=2)[0]
            cd_trellis = (dist1.mean() + dist2.mean()).item()
            
        except Exception as e:
            print(f"Error loading GLB {trellis_path}: {e}")
            logging.error(f"Batch {batch_idx + 1}: Error loading GLB {trellis_path}: {e}")
            cd_trellis = None

    if gt_path and os.path.exists(gt_path):
        try:
            with zipfile.ZipFile(gt_path, 'r') as zip_ref:
                stl_files = [f for f in zip_ref.namelist() if f.endswith('.stl')]
                if stl_files:
                    with zip_ref.open(stl_files[0]) as stl_file:
                        stl_data = io.BytesIO(stl_file.read())
                        gt_mesh = trimesh.load(stl_data, file_type='stl')
                        gt_points_np = trimesh.sample.sample_surface(gt_mesh, 10000)[0]
                        # 正規化：移動到原點，縮放到單位範圍
                        gt_points_np -= np.mean(gt_points_np, axis=0)
                        scale = np.max(np.abs(gt_points_np))
                        if scale > 1e-8:
                            gt_points_np /= scale
                        gt_points = torch.tensor(gt_points_np, dtype=torch.float32, device=device)
                        dist1 = torch.cdist(gen_points.unsqueeze(0), gt_points.unsqueeze(0)).min(dim=2)[0]
                        dist2 = torch.cdist(gt_points.unsqueeze(0), gen_points.unsqueeze(0)).min(dim=2)[0]
                        cd_stl = (dist1.mean() + dist2.mean()).item()
                        
        except Exception as e:
            print(f"Error loading STL {gt_path}: {e}")
            logging.error(f"Batch {batch_idx + 1}: Error loading STL {gt_path}: {e}")
            cd_stl = None

    # 總損失（加權組合）
    alpha = 0.1  # MSE 權重
    beta = 0.1   # LPIPS 權重
    gamma = 2.0  # TRELLIS Chamfer Distance 權重
    delta = 1.5  # STL Chamfer Distance 權重（可選）
    
    total_loss = alpha * mse_loss + beta * lpips_loss
    if cd_trellis is not None:
        total_loss += gamma * cd_trellis
    else:
        print(f"Warning: No valid TRELLIS GLB for batch {batch_idx + 1}")
        logging.warning(f"Batch {batch_idx + 1}: No valid TRELLIS GLB")
        return None, mse_loss, lpips_loss, cd_trellis, cd_stl
    if cd_stl is not None:
        total_loss += delta * cd_stl

    return total_loss, mse_loss, lpips_loss, cd_trellis, cd_stl
def train_distillation(model, train_loader, num_epochs=100, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    lpips_loss_fn = LPIPS(net='vgg').to(device)
    total_steps = len(train_loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps/5, eta_min=1e-5)

    if hasattr(model.backbone, 'gradient_checkpointing'):
        model.backbone.gradient_checkpointing = True
        print("Enabled gradient checkpointing to save memory.")

    loss_csv_path = "./distillation_loss_history.csv"
    os.makedirs(os.path.dirname(loss_csv_path), exist_ok=True)
    with open(loss_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'batch', 'mse_loss', 'lpips_loss', 'cd_trellis', 'cd_stl', 'total_loss', 'lr', 'global_step']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    global_step = 0
    for epoch in tqdm(range(num_epochs), desc="Epoch", total=num_epochs):
        model.train()
        total_loss_epoch = 0
        valid_batches = 0
        total_batches = len(train_loader)
        bar_length = 20

        for batch_idx, batch in enumerate(train_loader):
            if batch is None or not batch.get("valid", False):
                print(f"Skipping invalid batch {batch_idx + 1} due to loading error or invalid image")
                logging.warning(f"Batch {batch_idx + 1}: Skipped due to invalid batch")
                continue
            images = batch["image"].to(device)
            gt_path = batch["gt_path"][0] if isinstance(batch["gt_path"], list) else batch["gt_path"]
            trellis_path = batch["trellis_path"][0] if isinstance(batch["trellis_path"], list) else batch["trellis_path"]
            optimizer.zero_grad()

            try:
                scene_codes = model.forward(images, device)
            except Exception as e:
                print(f"Error in model forward for batch {batch_idx + 1}: {e}")
                logging.error(f"Batch {batch_idx + 1}: Error in model forward: {e}")
                continue

            result = compute_loss_and_metrics(
                model, scene_codes, images, gt_path, trellis_path, device, lpips_loss_fn, batch_idx, epoch
            )
            if result[0] is None:
                continue
            total_loss, mse_loss, lpips_loss, cd_trellis, cd_stl = result

            try:
                total_loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Error in backward/step for batch {batch_idx + 1}: {e}")
                logging.error(f"Batch {batch_idx + 1}: Error in backward/step: {e}")
                continue
            scheduler.step()
            global_step += 1

            total_loss_epoch += total_loss.item()
            valid_batches += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                with open(loss_csv_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1,
                        'mse_loss': mse_loss.item(),
                        'lpips_loss': lpips_loss.item(),
                        'cd_trellis': cd_trellis if cd_trellis is not None else 'N/A',
                        'cd_stl': cd_stl if cd_stl is not None else 'N/A',
                        'total_loss': total_loss.item(),
                        'lr': optimizer.param_groups[0]['lr'],
                        'global_step': global_step
                    })

            batch_progress = int((batch_idx + 1) * bar_length / total_batches)
            batch_bar = f"Batch {batch_idx+1}/{total_batches} [Epoch {epoch+1}/{num_epochs}]: |{'-' * batch_progress}{' ' * (bar_length - batch_progress)}| "
            cd_trellis_str = f"{cd_trellis:.4f}" if cd_trellis is not None else 'N/A'
            cd_stl_str = f"{cd_stl:.4f}" if cd_stl is not None else 'N/A'
            log_message = (
                f"{batch_bar}mse loss={mse_loss.item():.4f} lpips loss={lpips_loss.item():.4f} "
                f"cd_trellis={cd_trellis_str} cd_stl={cd_stl_str} total loss={total_loss.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} step={global_step}"
            )
            print(log_message, end="\n")
            logging.info(log_message)

            del scene_codes, total_loss, mse_loss, lpips_loss
            torch.cuda.empty_cache()

        if valid_batches > 0:
            avg_loss = total_loss_epoch / valid_batches
            log_message = (
                f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, "
                f"Valid Batches: {valid_batches}/{total_batches}, "
                f"Current LR: {optimizer.param_groups[0]['lr']:.6f}, Global Step: {global_step}"
            )
            print(f"\n{log_message}\n")
            logging.info(log_message)
        else:
            print(f"\nEpoch {epoch+1}/{num_epochs}, No valid batches\n")
            logging.warning(f"Epoch {epoch+1}: No valid batches")

    save_path = "./distillation_weights.ckpt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to '{save_path}'")
    print(f"Loss history saved to '{loss_csv_path}'")
    logging.info(f"Model weights saved to '{save_path}'")
    logging.info(f"Loss history saved to '{loss_csv_path}'")

if __name__ == "__main__":
    data_folder = "D:/Serena/TripoSR/lora_blender" # 訓練要用的圖片檔案位置 (資料夾)
    gt_folder = "D:/Serena/TripoSR/lora_data_gt"
    trellis_folder = "D:/Serena/TripoSR/teacher_train" # trellis 生成的模型檔案位置 (資料夾)

    dataset = PNGDataset(folder_path=data_folder, gt_folder=gt_folder, trellis_folder=trellis_folder, image_size=512)
    valid_indices = [i for i, (stl_folder, subfolder, _) in enumerate(dataset.image_files)
                     if subfolder in dataset.trellis_mapping]
    num_samples = int(len(valid_indices) * 0.5)
    indices = random.sample(valid_indices, num_samples)
    train_dataset = Subset(dataset, indices)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"Using {num_samples} samples (50% of {len(valid_indices)} PNGs with GLB)")
    logging.info(f"Using {num_samples} samples (50% of {len(valid_indices)} PNGs with GLB)")

    config_path = "D:/Serena/TripoSR/TripoSR/config.yaml"
    ckpt_path = "D:/Serena/TripoSR/TripoSR/model.ckpt"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    cfg.use_lora = False
    model = TSR(cfg=cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    print("Pretrained weights loaded successfully.")
    logging.info("Pretrained weights loaded successfully.")

    train_distillation(model, train_loader, num_epochs=100, lr=1e-4, device="cuda")
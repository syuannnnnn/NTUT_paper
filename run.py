import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import xatlas
import shutil

from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture
from omegaconf import OmegaConf
import trimesh
import zipfile
import open3d as o3d
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from pathlib import Path

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, nargs="+", help="Path to input image(s).")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="E:/Lab/Eassy/TripoSR/G_TripoSR_LoRA/TripoSR",
    # default= "stabilityai/TripoSR",
    type=str,
    help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'",
)
parser.add_argument(
    "--lora-weights",
    default="./lora_weights_test3.ckpt",
    type=str,
    help="Path to the trained LoRA weights."
)
parser.add_argument(
    "--chunk-size",
    default=8192,
    type=int,
    help="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 8192",
)
parser.add_argument(
    "--mc-resolution",
    default=192,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="output/",
    type=str,
    help="Output directory to save the results. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)
parser.add_argument(
    "--bake-texture",
    action="store_true",
    help="Bake a texture atlas for the extracted mesh, instead of vertex colors",
)
parser.add_argument(
    "--texture-resolution",
    default=2048,
    type=int,
    help="Texture atlas resolution, only useful with --bake-texture. Default: 2048"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)

parser.add_argument(
    "--ground-truth",
    type=str,
    default=None,
    help="Path to ground truth STL file for evaluation (e.g., .stl). Optional.",
)

parser.add_argument(
    "--align",
    action="store_true",
    help="If specified, align predicted point cloud to ground truth using ICP before evaluation.",
)

args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

timer.start("Initializing model")

config_path = os.path.join(args.pretrained_model_name_or_path, "config.yaml")
cfg = OmegaConf.load(config_path)
OmegaConf.resolve(cfg)
cfg.use_lora = True
cfg.lora_rank = 32
cfg.lora_alpha = 128

# 初始化模型，不直接加載權重
model = TSR(cfg=cfg)

# 加載原始權重，忽略 LoRA 參數
ckpt_path = os.path.join(args.pretrained_model_name_or_path, "model.ckpt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt, strict=False)
logging.info(f"Loaded base weights from {ckpt_path}")

# 加載 LoRA 權重
if args.lora_weights:
    lora_state_dict = torch.load(args.lora_weights, map_location="cpu", weights_only=True)
    model.load_state_dict(lora_state_dict, strict=False)
    logging.info(f"Loaded LoRA weights from {args.lora_weights}")

model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
timer.end("Initializing model")

# 處理圖像
timer.start("Processing images")
images = []

if args.no_remove_bg:
    rembg_session = None
else:
    rembg_session = rembg.new_session()

for i, image_path in enumerate(args.image):
    if args.no_remove_bg:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        if not os.path.exists(os.path.join(output_dir, str(i))):
            os.makedirs(os.path.join(output_dir, str(i)))
        image.save(os.path.join(output_dir, str(i), f"input.png"))
    images.append(image)
timer.end("Processing images")

# 推理每張圖像
for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    if args.render:
        timer.start("Rendering")
        
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
        save_video(render_images[0], os.path.join(output_dir, str(i), f"render.mp4"), fps=30)
        
        timer.end("Rendering")

    timer.start("Extracting mesh")
    meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
    timer.end("Extracting mesh")

    out_mesh_path = os.path.join(output_dir, str(i), f"mesh.{args.model_save_format}")

    if args.bake_texture:
        out_texture_path = os.path.join(output_dir, str(i), "texture.png")

        timer.start("Baking texture")
        bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
        timer.end("Baking texture")

        timer.start("Exporting mesh and texture")
        xatlas.export(out_mesh_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
        timer.end("Exporting mesh and texture")

    else:
        timer.start("Exporting mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")

    # 新增評估程式
    # 網格 正規化
    def normalize_mesh(mesh):
        mesh.vertices -= mesh.centroid  # 中心化 (扣掉質心)
        scale = np.max(np.abs(mesh.vertices)) # 找頂點座標最大值
        mesh.vertices /= scale # 縮放到 [-1, 1] 間

        return mesh, scale

    # 生成點雲並保存
    timer.start("Generating point cloud")
    mesh = meshes[0]

    # 正規化預測網格
    pred_mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    pred_mesh_trimesh, scale_pred = normalize_mesh(pred_mesh_trimesh)
    pred_points_np = trimesh.sample.sample_surface(pred_mesh_trimesh, count=10000)[0]
    pred_points = torch.tensor(pred_points_np, dtype=torch.float32, device=device)

    out_pointcloud_path = os.path.join(output_dir, str(i), "pointcloud.ply")
    trimesh.points.PointCloud(pred_points.cpu().numpy()).export(out_pointcloud_path)
    timer.end("Generating point cloud")

    # 如果提供了 ground truth STL 文件，計算 FS 和 CD
    if args.ground_truth:
        timer.start("Evaluating point cloud")

        # 解壓 ZIP 文件
        gt_path = args.ground_truth
        if gt_path.endswith('.zip'):
            with zipfile.ZipFile(gt_path, 'r') as zip_ref:
                # 處理 ZIP 檔案中的 STL 文件
                stl_files = [f for f in zip_ref.namelist() if f.endswith('.stl')]
                if not stl_files:
                    raise ValueError("No STL file found in the ZIP archive.")
                stl_file = stl_files[0]
                # 解壓縮 (暫存資料夾)
                temp_dir = os.path.join(output_dir, str(i), "temp")
                os.makedirs(temp_dir, exist_ok=True)
                zip_ref.extract(stl_file, temp_dir) # 提取壓縮
                gt_path = os.path.join(temp_dir, stl_file)

                # 刪除暫存資料夾
               #  shutil.rmtree(Path(temp_dir))
        else:
            gt_path = args.ground_truth  # 如果不是 ZIP，直接使用路徑

        # 處理 ground truth STL
        gt_mesh = trimesh.load(gt_path, file_type='stl')
        gt_mesh, scale_gt = normalize_mesh(gt_mesh)
        gt_points_np = trimesh.sample.sample_surface(gt_mesh, count=10000)[0] # np 
        # trimesh 輸出的是 numpy，但後續計算 chamfer_distance 接收 PyTorch 輸入
        gt_points = torch.tensor(gt_points_np, dtype=torch.float32, device=device) # PyTorch

        # 保存 ground truth 點雲
        gt_pointcloud_path = os.path.join(output_dir, str(i), "ground_truth_pointcloud.ply")
        trimesh.points.PointCloud(gt_points.cpu().numpy()).export(gt_pointcloud_path)
        logging.info(f"Ground truth point cloud saved to {gt_pointcloud_path}")

        # 網格正規化後的點雲
        gen_points = pred_points  
        logging.info(f"Predicted points range: {gen_points.min(dim=0)[0], gen_points.max(dim=0)[0]}")
        logging.info(f"Ground truth points range: {gt_points.min(dim=0)[0], gt_points.max(dim=0)[0]}")

        # ICP 對齊（正規化後）
        if args.align:
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(pred_points_np) # 轉成 o3d 的座標 (接受 np 格式)
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_points_np)

            threshold = 0.05 # ICP 的距離閥值
            trans_init = np.eye(4) # 處理平移旋轉
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pred_pcd, gt_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            pred_pcd.transform(reg_p2p.transformation)
            gen_points_np = np.asarray(pred_pcd.points) # 從 o3d 內部格式轉回 NumPy 陣列
            gen_points = torch.tensor(gen_points_np, dtype=torch.float32, device=device)
            logging.info("ICP alignment applied.")

        # 檢查點雲有效性
        if gen_points.shape[0] == 0 or gt_points.shape[0] == 0:
            raise ValueError("Empty point cloud detected")
        if torch.isnan(gen_points).any() or torch.isnan(gt_points).any():
            raise ValueError("NaN values detected in point cloud")
        logging.info(f"Pred points shape: {gen_points.shape}, range: {gen_points.min(dim=0)[0], gen_points.max(dim=0)[0]}")
        logging.info(f"GT points shape: {gt_points.shape}, range: {gt_points.min(dim=0)[0], gt_points.max(dim=0)[0]}")

        # 計算 CD 和 FS
        pred_pcl = Pointclouds(points=[gen_points])
        gt_pcl = Pointclouds(points=[gt_points])
        loss, _ = chamfer_distance(pred_pcl, gt_pcl, point_reduction=None, batch_reduction=None)
        dist1, dist2 = loss  # dist1 和 dist2 是形狀為 [1, N] 的張量
        cd = (dist1.mean() + dist2.mean()).item() # 取各自平均後的數值
        d = 0.1
        precision = (dist1 < d).float().mean().item()
        recall = (dist2 < d).float().mean().item()
        fs = 2 * (precision * recall) / (precision + recall + 1e-8)
        logging.info(f"Chamfer Distance: {cd:.6f}")
        logging.info(f"F-Score (tau=0.1): {fs:.6f}")       

        # 如果解壓了 STL 文件，清理臨時文件
        if args.ground_truth.endswith('.zip') and os.path.exists(gt_path):
            os.remove(gt_path)

        timer.end("Evaluating point cloud")
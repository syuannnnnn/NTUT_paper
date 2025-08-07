import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import csv
import os

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def normalize_mesh(mesh):
    vertices = mesh.vertices
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    scale = np.max(np.abs(vertices))
    if scale > 0:
        vertices = vertices / scale
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), scale

def compute_loss_and_metrics(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, device: str, num_points: int = 10000):
    # 使用 trimesh 採樣點雲，對齊 Trellis 的 10,000 點
    pred_points_np = trimesh.sample.sample_surface(pred_mesh, count=num_points)[0]
    gt_points_np = trimesh.sample.sample_surface(gt_mesh, count=num_points)[0]
    pred_points = torch.tensor(pred_points_np, dtype=torch.float32, device=device)
    gt_points = torch.tensor(gt_points_np, dtype=torch.float32, device=device)
    
    # 計算 Chamfer Distance
    pred_points = pred_points.unsqueeze(0)  # [1, num_points, 3]
    gt_points = gt_points.unsqueeze(0)     # [1, num_points, 3]
    dist1 = torch.cdist(pred_points, gt_points).min(dim=2)[0]  # [1, num_points]
    dist2 = torch.cdist(gt_points, pred_points).min(dim=2)[0]  # [1, num_points]
    cd = (dist1.mean() + dist2.mean())  # 標量
    
    # 計算 F-Score，閾值 0.1 對齊 Trellis
    threshold = 0.1
    precision = (dist1 < threshold).float().mean().item()
    recall = (dist2 < threshold).float().mean().item()
    fs = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return cd, fs, precision, recall

def train_lora(model, train_loader, num_epochs=300, lr=5e-4, device="cuda", num_points=10000):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs("./", exist_ok=True)
    csv_file = open("./loss_history.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'total_loss', 'cd', 'fs', 'precision', 'recall'])

    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
    ])

    for epoch in range(num_epochs):
        model.train()
        total_loss_epoch = 0.0
        total_cd = 0.0
        total_fs = 0.0
        total_precision = 0.0
        total_recall = 0.0
        num_batches = 0

        for batch in train_loader:
            images, gt_meshes = batch
            images = images.to(device)
            batch_size = images.shape[0]
            logging.info(f"Batch size: {batch_size}, Image shape: {images.shape}")

            images = torch.stack([transform(img) for img in images])

            optimizer.zero_grad()
            with torch.no_grad():
                # 確保批次大小一致
                scene_codes = model([images], device=device)
                if len(scene_codes) != batch_size:
                    logging.warning(f"Scene codes batch size {len(scene_codes)} does not match images batch size {batch_size}")
                    scene_codes = scene_codes[:batch_size]

            meshes = model.extract_mesh(scene_codes, resolution=256)

            pred_points_list = []
            gt_points_list = []
            for i, pred_mesh in enumerate(meshes):
                pred_mesh_trimesh = trimesh.Trimesh(vertices=pred_mesh.vertices, faces=pred_mesh.faces)
                pred_mesh_trimesh, _ = normalize_mesh(pred_mesh_trimesh)
                pred_points_list.append(pred_mesh_trimesh)

                gt_mesh = gt_meshes[i]
                gt_mesh, _ = normalize_mesh(gt_mesh)
                gt_points_list.append(gt_mesh)

            # 僅使用第一個網格進行損失計算（模擬批次大小 1）
            cd, fs, precision, recall = compute_loss_and_metrics(pred_points_list[0], gt_points_list[0], device, num_points)
            
            # 計算圖像損失，確保形狀匹配
            total_loss = model.compute_loss(images[:1], scene_codes[:1])  
            if total_loss.shape != torch.Size([]):
                logging.warning(f"Unexpected loss shape: {total_loss.shape}")
                total_loss = total_loss.mean()

            total_loss_with_cd = total_loss + 10 * cd

            for pred_mesh in meshes:
                vertices = torch.tensor(pred_mesh.vertices, dtype=torch.float32, device=device)
                ranges = torch.max(vertices, dim=0)[0] - torch.min(vertices, dim=0)[0]
                range_loss = torch.abs(ranges - ranges.mean()).mean()
                total_loss_with_cd += 0.1 * range_loss

            total_loss_with_cd.backward()
            optimizer.step()

            total_loss_epoch += total_loss_with_cd.item()
            total_cd += cd.item()
            total_fs += fs
            total_precision += precision
            total_recall += recall
            num_batches += 1

        avg_loss = total_loss_epoch / num_batches
        avg_cd = total_cd / num_batches
        avg_fs = total_fs / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}, CD: {avg_cd:.6f}, F-Score: {avg_fs:.6f}, Precision: {avg_precision:.6f}, Recall: {avg_recall:.6f}")
        csv_writer.writerow([epoch + 1, avg_loss, avg_cd, avg_fs, avg_precision, avg_recall])

    csv_file.close()
    torch.save(model.state_dict(), "./lora_weights_final.ckpt")
    logging.info("Training completed, weights saved to ./lora_weights_final.ckpt")
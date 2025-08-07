import math
import os
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
)


class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

        # 新增 LoRA 相關配置
        use_lora: bool = False
        lora_rank: int = 4
        lora_alpha: float = 1.0

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str,
        use_lora: bool = False, lora_rank: int = 4, lora_alpha: float = 1.0
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        # ----------------------
        # 添加 LoRA 配置
        cfg.use_lora = use_lora
        cfg.lora_rank = lora_rank
        cfg.lora_alpha = lora_alpha
        # ---------------------
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt)
        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        # self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        # 將 LoRA 參數傳遞給 backbone
        self.backbone = find_class(self.cfg.backbone_cls)(
            OmegaConf.merge(
                self.cfg.backbone,
                {
                    "use_lora": self.cfg.use_lora,
                    "lora_rank": self.cfg.lora_rank,
                    "lora_alpha": self.cfg.lora_alpha,
                }
            )
        )
        # ---------------------
        
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    def forward(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(
            device
        )
        # print(f"rgb_cond shape after image_processor: {rgb_cond.shape}")  # 添加形狀檢查

        
        # 修復形狀：移除多餘的維度
        # 假設 [1, 1, 1, 512, 512, 512] 應為 [1, 512, 512, 3] 或類似

        # ------------------------
            # while len(rgb_cond.shape) > 4:
            #     rgb_cond = rgb_cond.squeeze(0)  # 從頭移除大小為 1 的維度
            # print(f"rgb_cond shape after squeezing: {rgb_cond.shape}")

            # # 確保形狀是 [B, H, W, C] 或 [B, C, H, W]
            # if len(rgb_cond.shape) == 4:
            #     if rgb_cond.shape[-1] in [1, 3]:  # [B, H, W, C]
            #         pass
            #     elif rgb_cond.shape[1] in [1, 3]:  # [B, C, H, W]
            #         rgb_cond = rgb_cond.permute(0, 2, 3, 1)  # [B, H, W, C]

            # # 添加 Nv 維度
            # rgb_cond = rgb_cond[:, None]  # [B, Nv, H, W, C]
            # print(f"rgb_cond shape after adding Nv: {rgb_cond.shape}")
        # ------------------------

        # 修復形狀(可以訓練版本)
        # if len(rgb_cond.shape) == 6 and rgb_cond.shape[:3] == (1, 1, 1):
        #     # [1, 1, 1, 512, 512, 512] -> [1, 512, 512, 3]
        #     rgb_cond = rgb_cond.squeeze(0).squeeze(0).squeeze(0)  # [512, 512, 512]
        #     rgb_cond = rgb_cond[..., :3]  # [512, 512, 3]
        #     rgb_cond = rgb_cond[None, ...]  # [1, 512, 512, 3]
        # elif len(rgb_cond.shape) == 4:
        #     if rgb_cond.shape[0] == 512:  # [512, 1, 512, 512]
        #         rgb_cond = rgb_cond.squeeze(1)  # [512, 512, 512]
        #         rgb_cond = rgb_cond[..., :3]  # [512, 512, 3]
        #         rgb_cond = rgb_cond[None, ...]  # [1, 512, 512, 3]
        #     elif rgb_cond.shape[1] == 3:  # [B, C, H, W]
        #         rgb_cond = rgb_cond.permute(0, 2, 3, 1)  # [B, H, W, C]

        # # 添加 Nv 維度
        # rgb_cond = rgb_cond[:, None]  # [B, Nv, H, W, C]
        # # print(f"rgb_cond shape after correction: {rgb_cond.shape}")

        # batch_size = rgb_cond.shape[0]
        # # print(f"rgb_cond shape after adding Nv: {rgb_cond.shape}")

        # input_image_tokens: torch.Tensor = self.image_tokenizer(
        #     rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        # )

        # input_image_tokens = rearrange(
        #     input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        # )

        # tokens: torch.Tensor = self.tokenizer(batch_size)

        # tokens = self.backbone(
        #     tokens,
        #     encoder_hidden_states=input_image_tokens,
        # )

        # scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        # return scene_codes
    # (可以訓練版本)---------------------------
       # 修復形狀：移除多餘的維度並處理通道數

        while len(rgb_cond.shape) > 4:
            rgb_cond = rgb_cond.squeeze(0)  # 移除大小為 1 的維度，直到 4 維或更低
        
        if len(rgb_cond.shape) == 3:  # [H, W, C]，單張圖像
            rgb_cond = rgb_cond[None, ...]  # [1, H, W, C]
        elif len(rgb_cond.shape) == 4:
            if rgb_cond.shape[-1] not in [1, 3]:  # [B, H, W, C']，C' 不正確
                rgb_cond = rgb_cond[..., :3]  # 強制轉為 RGB
            elif rgb_cond.shape[1] in [1, 3]:  # [B, C, H, W]
                rgb_cond = rgb_cond.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        rgb_cond = rgb_cond[:, None]  # [B, Nv, H, W, C]
        # print(f"rgb_cond shape after correction: {rgb_cond.shape}")
        
        batch_size = rgb_cond.shape[0]
        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),)
        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
        tokens: torch.Tensor = self.tokenizer(batch_size)
        tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))

        return scene_codes

    def render(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)

        return images

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, scene_codes, has_vertex_color, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            color = None
            if has_vertex_color:
                with torch.no_grad():
                    color = self.renderer.query_triplane(
                        self.decoder,
                        v_pos,
                        scene_code,
                    )["color"]
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy() if has_vertex_color else None,
            )
            meshes.append(mesh)
        return meshes

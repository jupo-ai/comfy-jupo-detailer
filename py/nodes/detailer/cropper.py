from comfy_api.latest import io
from ...utils import mk_name
from .common import PACKAGE_NAME, CATEGORY, IO_CROP_INFO

from . import prepare_utils
from . import crop_utils

import math
import torch
import comfy.model_management
import comfy.utils
from nodes import MAX_RESOLUTION


class DetailerCropper(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "DetailerCropper"), 
            display_name="Detailer Cropper", 
            category=CATEGORY, 
            inputs=[
                io.Image.Input("images"), 
                io.Mask.Input("masks"), 
                
                # mask
                io.Boolean.Input("mask_fill_holes", default=True, tooltip="マスクの小さな穴を塞ぐ。"), 
                io.Int.Input("mask_expand_pixels", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="マスクを拡張する。"), 
                io.Int.Input("mask_blend_pixels", default=4, min=0, max=64, step=1, tooltip="マスクを拡張・ブラーをかけて元画像と合成する。"), 
                io.Float.Input("mask_filter_threshold", default=0.1, min=0, max=1, step=0.01, tooltip="この値より低い部分はマスク無しになる。"), 

                # context
                io.Float.Input("context_extend_factor", default=1.2, min=1.0, max=100.0, step=0.01, tooltip="マスクから、コンテキスト領域を全方向に一定の倍率で拡大する。"), 

                # output
                io.Combo.Input("output_resize", options=["None", "keep aspect", "constant"], default="keep aspect", tooltip="出力サイズを指定する。\nNone: リサイズを行わない。ただし複数マスクの場合は1枚目のサイズに合わせる\nkeep aspect: アスペクト比を保ったまま output_resolution の解像度になるように調整(例: 1024の場合、面積が1024*1024に近くなるように)\nconstant: output_width, output_heightに強制"), 
                io.Combo.Input("resize_algorithm", options=["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"], default="bilinear"), 
                io.Int.Input("output_resolution", default=1024, min=0, step=64), 
                io.Int.Input("output_width", default=1024, min=64, max=MAX_RESOLUTION, step=1), 
                io.Int.Input("output_height", default=1024, min=64, max=MAX_RESOLUTION, step=1), 
                io.Int.Input("output_padding", default=16, min=8, step=8, tooltip="幅と高さがこの倍数になるように調整する"), 
            ], 
            outputs=[
                IO_CROP_INFO.Output(display_name="crop_info"), 
                io.Image.Output(display_name="cropped_images"), 
                io.Mask.Output(display_name="cropped_masks"), 
            ], 
        )

    @staticmethod
    def calculate_keep_aspect_size(width: int, height: int, resolution: int):
        if width <= 0 or height <= 0:
            return width, height
        if resolution <= 0:
            return width, height

        target_area = resolution * resolution
        scale = math.sqrt(target_area / (width * height))
        target_w = max(1, round(width * scale))
        target_h = max(1, round(height * scale))
        return target_w, target_h
    
    @classmethod
    def execute(
        cls, 
        images: torch.Tensor, 
        masks: torch.Tensor, 
        
        mask_fill_holes: bool = True, 
        mask_expand_pixels: int = 0, 
        mask_blend_pixels: int = 4, 
        mask_filter_threshold: float = 0.1, 
        
        context_extend_factor: float = 1.2, 
        
        output_resize: str = "keep aspect", 
        resize_algorithm: str = "bilinear", 
        output_resolution: int = 1024, 
        output_width: int = 1024, 
        output_height: int = 1024, 
        output_padding: int = 16, 
    ):

        # 前準備
        images = prepare_utils.normalize_image_tensor(images).clone()
        masks = prepare_utils.prepare_mask(masks, images)

        input_image_batch_size = images.shape[0]
        images, masks = prepare_utils.sync_batch_size(images, masks)

        device = comfy.model_management.get_torch_device()
        images = images.to(device)
        masks = masks.to(device)
        
        # 実行
        output_info = []
        output_images = []
        output_masks = []

        batch_size = images.shape[0]
        pbar = comfy.utils.ProgressBar(batch_size)
        
        for i in range(batch_size):
            sub_image = images[i:i+1]
            sub_mask = masks[i:i+1]
            
            # --- マスク処理 ---
            # マスクの穴埋め
            if mask_fill_holes:
                sub_mask = crop_utils.fill_mask_holes(sub_mask)
            
            # マスクの拡大
            if mask_expand_pixels > 0:
                sub_mask = crop_utils.expand_mask(sub_mask, mask_expand_pixels)
            
            # マスク縁をブラー
            if mask_blend_pixels > 0:
                sub_mask = crop_utils.expand_mask(sub_mask, mask_blend_pixels)
                sub_mask = crop_utils.blur_mask(sub_mask, mask_blend_pixels)
            
            # 閾値以下をフィルタリング
            if mask_filter_threshold > 0:
                sub_mask = crop_utils.filter_mask_high_pass(sub_mask, mask_filter_threshold)
            
            
            # --- コンテキストエリア ---
            # マスクからコンテキストエリアを取得
            bx, by, bw, bh = crop_utils.find_context_area(sub_mask)

            # コンテキストエリアを拡大
            if context_extend_factor > 1.0:
                bx, by, bw, bh = crop_utils.expand_context_area(sub_mask, bx, by, bw, bh, context_extend_factor)
            
            # コンテキステリアが存在しない場合に、元画像のサイズにフォールバック
            if bx[0] == -1:
                bx[0] = 0
                by[0] = 0
                bw[0] = sub_image.shape[2]
                bh[0] = sub_image.shape[1]
            
            
            # --- クロップ ---
            crop_x = bx[0].item()
            crop_y = by[0].item()
            crop_w = bw[0].item()
            crop_h = bh[0].item()
            
            crop_padding = output_padding
            resize_output = True
            if output_resize == "None":
                target_w = crop_w
                target_h = crop_h
                resize_output = False
            elif output_resize == "keep aspect":
                target_w, target_h = cls.calculate_keep_aspect_size(crop_w, crop_h, output_resolution)
            elif output_resize == "constant":
                target_w = output_width
                target_h = output_height
            else:
                raise ValueError(f"Unknown output_resize mode: {output_resize}")

            # 2枚目以降は1枚目の出力サイズに合わせる
            if i > 0:
                first_image = output_images[0]
                target_w = first_image.shape[1]
                target_h = first_image.shape[0]
                crop_padding = 0
                resize_output = True
            
            canvas_image, canvas_mask, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h \
                = crop_utils.advanced_crop(sub_image, sub_mask, crop_x, crop_y, crop_w, crop_h, target_w, target_h, crop_padding, resize_algorithm, resize_output)
            
            
            # --- 結果集計 ---
            info = {
                "canvas_to_orig": [cto_x, cto_y, cto_w, cto_h], 
                "canvas_image": canvas_image.cpu(), 
                "canvas_mask": canvas_mask.cpu(), 
                "cropped_to_canvas": [ctc_x, ctc_y, ctc_w, ctc_h], 
                "source_image_index": 0 if input_image_batch_size == 1 else i, 
            }
            
            output_images.append(cropped_image.squeeze(0).cpu())
            output_masks.append(cropped_mask.squeeze(0).cpu())
            output_info.append(info)
            pbar.update(1)
        
        cropped_masks = torch.stack(output_masks, dim=0)
        cropped_images = torch.stack(output_images, dim=0)
        
        return io.NodeOutput(output_info, cropped_images, cropped_masks)


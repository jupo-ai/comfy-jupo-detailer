from comfy_api.latest import io
from ...utils import mk_name
from .common import PACKAGE_NAME, CATEGORY, IO_CROP_INFO

from . import crop_utils
from . import prepare_utils

import math
import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
from nodes import MAX_RESOLUTION


class DetailerOutpainter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "DetailerOutpainter"),
            display_name="Detailer Outpainter",
            category=CATEGORY,
            inputs=[
                io.Image.Input("images"),

                io.Int.Input("extend_up_pixels", default=0, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("extend_down_pixels", default=0, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("extend_left_pixels", default=0, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("extend_right_pixels", default=0, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("mask_blend_pixels", default=4, min=0, max=64, step=1),

                io.Combo.Input(
                    "output_resize",
                    options=["None", "keep aspect", "constant"],
                    default="keep aspect",
                    tooltip="Detailerへ渡す出力サイズを指定する。Noneの場合も複数画像では1枚目のサイズに合わせる。",
                ),
                io.Combo.Input("resize_algorithm", options=["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"], default="bilinear"),
                io.Int.Input("output_resolution", default=1024, min=0, step=64),
                io.Int.Input("output_width", default=1024, min=64, max=MAX_RESOLUTION, step=1),
                io.Int.Input("output_height", default=1024, min=64, max=MAX_RESOLUTION, step=1),
                io.Int.Input("output_padding", default=16, min=8, step=8, tooltip="幅と高さがこの倍数になるように調整する"),
            ],
            outputs=[
                IO_CROP_INFO.Output(display_name="crop_info"),
                io.Image.Output(display_name="outpaint_images"),
                io.Mask.Output(display_name="outpaint_masks"),
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

    @staticmethod
    def expand_image_for_outpaint(
        image: torch.Tensor,
        up: int,
        down: int,
        left: int,
        right: int,
    ):
        """
        画像を上下左右へreplicateで拡張し、拡張部分だけ1.0のマスクを作る。
        入力/出力画像は [1, H, W, C]、マスクは [1, H, W]。
        """
        if min(up, down, left, right) < 0:
            raise ValueError("Outpaint extension pixels must be >= 0")

        b, h, w, c = image.shape
        image_nchw = image.permute(0, 3, 1, 2)
        expanded = F.pad(image_nchw, (left, right, up, down), mode="replicate")
        expanded = expanded.permute(0, 2, 3, 1)

        expanded_h = h + up + down
        expanded_w = w + left + right
        mask = torch.ones((b, expanded_h, expanded_w), dtype=image.dtype, device=image.device)
        mask[:, up:up + h, left:left + w] = 0.0

        return expanded, mask

    @staticmethod
    def resize_output_pair(
        image: torch.Tensor,
        mask: torch.Tensor,
        target_w: int,
        target_h: int,
        output_padding: int,
        resize_algorithm: str,
        resize_output: bool,
    ):
        if not resize_output:
            return image, mask

        target_w = crop_utils.pad_to_multiple(target_w, output_padding)
        target_h = crop_utils.pad_to_multiple(target_h, output_padding)
        target_size = (target_h, target_w)
        image = crop_utils.resize_image(image, target_size, resize_algorithm)
        mask = crop_utils.resize_mask(mask, target_size, resize_algorithm)
        return image, mask

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        extend_up_pixels: int = 0,
        extend_down_pixels: int = 0,
        extend_left_pixels: int = 0,
        extend_right_pixels: int = 0,
        mask_blend_pixels: int = 4,
        output_resize: str = "keep aspect",
        resize_algorithm: str = "bilinear",
        output_resolution: int = 1024,
        output_width: int = 1024,
        output_height: int = 1024,
        output_padding: int = 16,
    ):
        images = prepare_utils.normalize_image_tensor(images).clone()

        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        output_info = []
        output_images = []
        output_masks = []

        batch_size = images.shape[0]
        pbar = comfy.utils.ProgressBar(batch_size)

        for i in range(batch_size):
            sub_image = images[i:i + 1]
            expanded_image, expanded_mask = cls.expand_image_for_outpaint(
                sub_image,
                int(extend_up_pixels),
                int(extend_down_pixels),
                int(extend_left_pixels),
                int(extend_right_pixels),
            )

            if mask_blend_pixels > 0 and expanded_mask.any():
                expanded_mask = crop_utils.blur_mask(expanded_mask, int(mask_blend_pixels))

            expanded_h = expanded_image.shape[1]
            expanded_w = expanded_image.shape[2]

            resize_output = True
            if output_resize == "None":
                target_w = expanded_w
                target_h = expanded_h
                resize_output = False
            elif output_resize == "keep aspect":
                target_w, target_h = cls.calculate_keep_aspect_size(expanded_w, expanded_h, int(output_resolution))
            elif output_resize == "constant":
                target_w = int(output_width)
                target_h = int(output_height)
            else:
                raise ValueError(f"Unknown output_resize mode: {output_resize}")

            if i > 0:
                first_image = output_images[0]
                target_w = first_image.shape[1]
                target_h = first_image.shape[0]
                resize_output = True
                output_padding_for_item = 0
            else:
                output_padding_for_item = int(output_padding)

            output_image, output_mask = cls.resize_output_pair(
                expanded_image,
                expanded_mask,
                target_w,
                target_h,
                output_padding_for_item,
                resize_algorithm,
                resize_output,
            )

            info = {
                "canvas_to_orig": [0, 0, expanded_w, expanded_h],
                "canvas_image": expanded_image.cpu(),
                "canvas_mask": expanded_mask.cpu(),
                "cropped_to_canvas": [0, 0, expanded_w, expanded_h],
                "source_image_index": i,
                "outpaint_padding": [
                    int(extend_up_pixels),
                    int(extend_down_pixels),
                    int(extend_left_pixels),
                    int(extend_right_pixels),
                ],
            }

            output_images.append(output_image.squeeze(0).cpu())
            output_masks.append(output_mask.squeeze(0).cpu())
            output_info.append(info)
            pbar.update(1)

        return io.NodeOutput(
            output_info,
            torch.stack(output_images, dim=0),
            torch.stack(output_masks, dim=0),
        )

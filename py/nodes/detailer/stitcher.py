from comfy_api.latest import io
from ...utils import mk_name
from .common import PACKAGE_NAME, CATEGORY, IO_CROP_INFO

from . import prepare_utils
from . import stitch_utils

import torch
import comfy.model_management
import comfy.utils


class DetailerStitcher(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "DetailerStitcher"),
            display_name="Detailer Stitcher",
            category=CATEGORY,
            inputs=[
                IO_CROP_INFO.Input("crop_info"),
                io.Image.Input("inpainted"),
                io.Boolean.Input(
                    "override",
                    default=True,
                    tooltip="同じ元画像から複数クロップした結果を1枚の画像へ順番に戻す。",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @staticmethod
    def _get_inpainted(inpainted: torch.Tensor, index: int):
        if inpainted.shape[0] == 1:
            return inpainted[0:1]
        return inpainted[index:index + 1]

    @classmethod
    def execute(cls, crop_info: list[dict], inpainted: torch.Tensor, override: bool = True):
        if not crop_info:
            raise ValueError("crop_info is empty")

        inpainted = prepare_utils.normalize_image_tensor(inpainted).clone()
        if inpainted.shape[0] not in {1, len(crop_info)}:
            raise ValueError(
                f"Batch size mismatch: crop_info({len(crop_info)}), inpainted({inpainted.shape[0]})"
            )

        device = comfy.model_management.get_torch_device()
        inpainted = inpainted.to(device)

        pbar = comfy.utils.ProgressBar(total=len(crop_info))

        if not override:
            results = []
            for i, info in enumerate(crop_info):
                result, _ = stitch_utils.stitch_crop(
                    cls._move_info_to_device(info, device),
                    cls._get_inpainted(inpainted, i),
                )
                results.append(result.squeeze(0).cpu())
                pbar.update(1)
            return io.NodeOutput(torch.stack(results, dim=0))

        grouped_images = {}
        grouped_order = []

        for i, info in enumerate(crop_info):
            info = cls._move_info_to_device(info, device)
            source_index = int(info.get("source_image_index", i))
            if source_index not in grouped_images:
                grouped_images[source_index] = stitch_utils.crop_canvas_to_original(
                    info["canvas_image"],
                    info["canvas_to_orig"],
                )
                grouped_order.append(source_index)

            canvas = stitch_utils.put_original_on_canvas(
                info["canvas_image"],
                grouped_images[source_index],
                info["canvas_to_orig"],
            )
            result, _ = stitch_utils.stitch_crop(
                info,
                cls._get_inpainted(inpainted, i),
                base_canvas=canvas,
            )
            grouped_images[source_index] = result
            pbar.update(1)

        results = []
        for source_index in grouped_order:
            results.append(grouped_images[source_index].squeeze(0).cpu())

        return io.NodeOutput(torch.stack(results, dim=0))

    @staticmethod
    def _move_info_to_device(info: dict, device: torch.device):
        moved = dict(info)
        moved["canvas_image"] = moved["canvas_image"].to(device)
        moved["canvas_mask"] = moved["canvas_mask"].to(device)
        return moved

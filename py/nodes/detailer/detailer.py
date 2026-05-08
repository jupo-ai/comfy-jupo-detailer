from comfy_api.latest import io
from ...utils import mk_name
from .common import PACKAGE_NAME, CATEGORY, IO_CROP_INFO

import comfy.samplers
import comfy.utils
import node_helpers
import torch
import torch.nn.functional as F
from nodes import common_ksampler


class Detailer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "Detailer"),
            display_name="Detailer",
            category=CATEGORY,
            is_input_list=True,
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                IO_CROP_INFO.Input("crop_info"),
                io.Image.Input("cropped_images"),
                io.Mask.Input("cropped_masks"),

                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("noise_mask", default=True),
            ],
            outputs=[
                IO_CROP_INFO.Output(display_name="crop_info"),
                io.Image.Output(display_name="inpainted"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        vae,
        positive,
        negative,
        crop_info: list[dict],
        cropped_images: torch.Tensor,
        cropped_masks: torch.Tensor,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        noise_mask: bool = True,
    ):
        model = cls._unwrap_single(model)
        vae = cls._unwrap_single(vae)
        seed = cls._unwrap_single(seed)
        steps = cls._unwrap_single(steps)
        cfg = cls._unwrap_single(cfg)
        sampler_name = cls._unwrap_single(sampler_name)
        scheduler = cls._unwrap_single(scheduler)
        denoise = cls._unwrap_single(denoise)
        noise_mask = cls._unwrap_single(noise_mask)
        cropped_images = cls._normalize_image_input(cropped_images)
        cropped_masks = cls._normalize_mask_input(cropped_masks)
        crop_info = cls._normalize_crop_info(crop_info)

        item_count = min(len(crop_info), cropped_images.shape[0], cropped_masks.shape[0])
        if item_count <= 0:
            raise ValueError("Detailer has no crop items to process.")

        cls._validate_conditioning_count("positive", positive, item_count)
        cls._validate_conditioning_count("negative", negative, item_count)

        outputs = []
        pbar = comfy.utils.ProgressBar(item_count)
        for i in range(item_count):
            one_image = cropped_images[i:i + 1].clone()
            one_mask = cropped_masks[i:i + 1].clone()
            item_positive = cls._select_conditioning(positive, i, item_count)
            item_negative = cls._select_conditioning(negative, i, item_count)

            inpaint_positive, inpaint_negative, latent = cls._inpaint_conditioning(
                item_positive,
                item_negative,
                one_image,
                vae,
                one_mask,
                noise_mask,
            )

            sampled_latent, = common_ksampler(
                model,
                int(seed) + i,
                steps,
                cfg,
                sampler_name,
                scheduler,
                inpaint_positive,
                inpaint_negative,
                latent,
                denoise=denoise,
            )

            outputs.append(cls._decode_latent(vae, sampled_latent).squeeze(0).cpu())
            pbar.update(1)

        return io.NodeOutput(crop_info[:item_count], torch.stack(outputs, dim=0))

    @staticmethod
    def _unwrap_single(value):
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    @classmethod
    def _normalize_crop_info(cls, crop_info):
        crop_info = cls._unwrap_single(crop_info)
        if not isinstance(crop_info, list):
            raise ValueError(f"Expected crop_info list, got {type(crop_info)}")
        return crop_info

    @classmethod
    def _normalize_image_input(cls, images):
        if isinstance(images, list):
            images = [img for img in images if torch.is_tensor(img)]
            if not images:
                raise ValueError("cropped_images is empty")
            images = torch.cat(images, dim=0)
        if not torch.is_tensor(images) or images.dim() != 4:
            raise ValueError(f"Expected cropped_images [B, H, W, C], got {type(images)}")
        return images

    @classmethod
    def _normalize_mask_input(cls, masks):
        if isinstance(masks, list):
            masks = [mask for mask in masks if torch.is_tensor(mask)]
            if not masks:
                raise ValueError("cropped_masks is empty")
            masks = torch.cat([mask.unsqueeze(0) if mask.dim() == 2 else mask for mask in masks], dim=0)
        if not torch.is_tensor(masks):
            raise ValueError(f"Expected cropped_masks tensor, got {type(masks)}")
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        if masks.dim() != 3:
            raise ValueError(f"Expected cropped_masks [B, H, W], got shape: {masks.shape}")
        return masks

    @classmethod
    def _select_conditioning(cls, conditioning, index: int, item_count: int):
        conditioning = cls._unwrap_conditioning_input(conditioning)
        if cls._is_conditioning_batch(conditioning, item_count):
            return conditioning[index]
        return conditioning

    @classmethod
    def _validate_conditioning_count(cls, name: str, conditioning, item_count: int):
        conditioning = cls._unwrap_conditioning_input(conditioning)
        if cls._is_conditioning_batch(conditioning, item_count):
            return
        if not cls._is_conditioning(conditioning):
            raise ValueError(f"{name} must be CONDITIONING or a list of CONDITIONING.")

    @classmethod
    def _unwrap_conditioning_input(cls, conditioning):
        if (
            isinstance(conditioning, list)
            and len(conditioning) == 1
            and not cls._is_conditioning(conditioning)
            and cls._is_conditioning(conditioning[0])
        ):
            return conditioning[0]
        return conditioning

    @classmethod
    def _is_conditioning_batch(cls, value, item_count: int):
        return (
            isinstance(value, list)
            and len(value) == item_count
            and all(cls._is_conditioning(item) for item in value)
        )

    @classmethod
    def _is_conditioning(cls, value):
        if not isinstance(value, list) or len(value) == 0:
            return False
        return all(cls._is_conditioning_entry(item) for item in value)

    @staticmethod
    def _is_conditioning_entry(value):
        return isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], dict)

    @classmethod
    def _inpaint_conditioning(cls, positive, negative, pixels, vae, mask, noise_mask=True):
        downscale_ratio = vae.spacial_compression_encode()
        height = (pixels.shape[1] // downscale_ratio) * downscale_ratio
        width = (pixels.shape[2] // downscale_ratio) * downscale_ratio
        mask = F.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(pixels.shape[1], pixels.shape[2]),
            mode="bilinear",
        )

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != height or pixels.shape[2] != width:
            y_offset = (pixels.shape[1] % downscale_ratio) // 2
            x_offset = (pixels.shape[2] % downscale_ratio) // 2
            pixels = pixels[:, y_offset:height + y_offset, x_offset:width + x_offset, :]
            mask = mask[:, :, y_offset:height + y_offset, x_offset:width + x_offset]

        masked_pixels = pixels.clone()
        keep_mask = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            masked_pixels[:, :, :, i] -= 0.5
            masked_pixels[:, :, :, i] *= keep_mask
            masked_pixels[:, :, :, i] += 0.5

        concat_latent = vae.encode(masked_pixels[:, :, :, :3])
        orig_latent = vae.encode(orig_pixels[:, :, :, :3])

        out_latent = {"samples": orig_latent}
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            out.append(node_helpers.conditioning_set_values(conditioning, {
                "concat_latent_image": concat_latent,
                "concat_mask": mask,
            }))
        return out[0], out[1], out_latent

    @staticmethod
    def _decode_latent(vae, samples):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

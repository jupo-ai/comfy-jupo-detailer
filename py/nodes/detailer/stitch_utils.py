import torch

from . import crop_utils


def _match_patch_channels(patch: torch.Tensor, base: torch.Tensor):
    base_channels = base.shape[-1]
    patch_channels = patch.shape[-1]

    if patch_channels == base_channels:
        return patch

    if patch_channels > base_channels:
        return patch[..., :base_channels]

    if patch_channels == 1 and base_channels >= 3:
        patch = patch.expand(*patch.shape[:-1], 3)
        patch_channels = patch.shape[-1]

    if patch_channels < base_channels:
        extra = base[..., patch_channels:]
        patch = torch.cat([patch, extra], dim=-1)

    return patch


def stitch_to_canvas(
    canvas_image: torch.Tensor,
    canvas_mask: torch.Tensor,
    inpainted: torch.Tensor,
    cropped_to_canvas: list[int],
    resize_algorithm: str = "bilinear",
):
    """
    inpaintedをcanvas座標へ戻し、canvas_maskで合成する。
    入力は [1, H, W, C] / [1, H, W] を想定する。
    """
    ctc_x, ctc_y, ctc_w, ctc_h = [int(v) for v in cropped_to_canvas]

    canvas = canvas_image.clone()
    region = canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]
    mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w].clamp(0.0, 1.0)

    patch = inpainted
    if patch.shape[1] != ctc_h or patch.shape[2] != ctc_w:
        patch = crop_utils.resize_image(patch, (ctc_h, ctc_w), resize_algorithm)
    patch = _match_patch_channels(patch, region).to(device=region.device, dtype=region.dtype)
    mask = mask.to(device=region.device, dtype=region.dtype).unsqueeze(-1)

    canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = patch * mask + region * (1.0 - mask)
    return canvas


def crop_canvas_to_original(canvas_image: torch.Tensor, canvas_to_orig: list[int]):
    orig_x, orig_y, orig_w, orig_h = [int(v) for v in canvas_to_orig]
    return canvas_image[:, orig_y:orig_y + orig_h, orig_x:orig_x + orig_w, :]


def put_original_on_canvas(canvas_image: torch.Tensor, original_image: torch.Tensor, canvas_to_orig: list[int]):
    orig_x, orig_y, orig_w, orig_h = [int(v) for v in canvas_to_orig]
    canvas = canvas_image.clone()
    original = original_image
    if original.shape[1] != orig_h or original.shape[2] != orig_w:
        original = crop_utils.resize_image(original, (orig_h, orig_w), "bilinear")
    original = _match_patch_channels(
        original,
        canvas[:, orig_y:orig_y + orig_h, orig_x:orig_x + orig_w, :],
    ).to(device=canvas.device, dtype=canvas.dtype)
    canvas[:, orig_y:orig_y + orig_h, orig_x:orig_x + orig_w, :] = original
    return canvas


def stitch_crop(
    crop_info: dict,
    inpainted: torch.Tensor,
    base_canvas: torch.Tensor | None = None,
    resize_algorithm: str = "bilinear",
):
    canvas_image = base_canvas if base_canvas is not None else crop_info["canvas_image"]
    canvas = stitch_to_canvas(
        canvas_image,
        crop_info["canvas_mask"],
        inpainted,
        crop_info["cropped_to_canvas"],
        resize_algorithm,
    )
    return crop_canvas_to_original(canvas, crop_info["canvas_to_orig"]), canvas

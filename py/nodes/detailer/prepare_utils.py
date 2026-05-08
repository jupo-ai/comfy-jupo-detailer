import torch
import torch.nn.functional as F

def normalize_image_tensor(image: torch.Tensor):
    """
    画像テンソルを [B, H, W, 3] の形に正規化
    """
    if not torch.is_tensor(image):
        raise TypeError(f"Expected image torch.Tensor, got {type(image)}")
    
    if image.dim() != 4:
        raise ValueError(f"Expected image tensor [B, H, W, C], got {image.dim()}D (shape: {image.shape})")

    # グレースケール(1ch) をRGB(3ch) に拡張
    # この時点で [B, H, W, C] の形になっている
    if image.shape[-1] == 1:
        image = image.expand(*image.shape[:-1], 3)
    elif image.shape[-1] not in {3, 4}:
        raise ValueError(f"Expected image channels 1, 3, or 4, got shape: {image.shape}")
    
    return image


def normalize_mask_tensor(mask: torch.Tensor):
    """
    マスクテンソルを [B, H, W] の形に正規化
    """
    if not torch.is_tensor(mask):
        raise TypeError(f"Expected mask torch.Tensor, got {type(mask)}")

    dims = mask.dim()
    if dims not in [2, 3]:
        raise ValueError(f"Expected mask tensor [B, H, W] or [H, W], got {dims}D (shape: {mask.shape})")

    # 2次元: [H, W] -> [1, H, W]
    if dims == 2:
        mask = mask.unsqueeze(0)

    # 浮動小数点数型に統一
    if not mask.is_floating_point():
        mask = mask.float()

    return mask
    

def fix_mask_shape(mask: torch.Tensor, image: torch.Tensor):
    """
    マスク解像度を画像サイズに合わせる。
    空マスクはゼロテンソルを作り直し、非空マスクは補間する。
    """
    # image は [B, H, W, C], mask は [B, H, W] と想定
    img_b, img_h, img_w = image.shape[0], image.shape[1], image.shape[2]
    mask_b, mask_h, mask_w = mask.shape[0], mask.shape[1], mask.shape[2]

    # バッチサイズが互換性(一致、もしくはどちらかが1)であるか確認
    batch_compatible = (mask_b == img_b) or mask_b == 1 or img_b == 1

    if batch_compatible:
        # 解像度が異なる場合のみ
        if mask_h != img_h or mask_w != img_w:
            if not mask.any():
                mask = torch.zeros(
                    (mask_b, img_h, img_w), 
                    device=image.device, 
                    dtype=image.dtype
                )
            else:
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
    return mask


def prepare_mask(mask: torch.Tensor, image: torch.Tensor):
    if mask is None:
        return torch.zeros(
            image.shape[:3], 
            dtype=image.dtype
        )
    
    m = normalize_mask_tensor(mask)
    m = fix_mask_shape(m, image)
    return m.to(dtype=image.dtype, device=image.device).clone()


def sync_batch_size(image: torch.Tensor, mask: torch.Tensor):
    """
    画像とマスクのバッチサイズを同期する
    """
    # 1. 現在のそれぞれの枚数（バッチサイズ）を確認
    img_b = image.shape[0]
    mask_b = mask.shape[0]

    # 2. 最終的に合わせるべき「最大枚数」を決定する
    # 例: imageが1枚、maskが10枚なら、target_bは10になる
    target_b = max(img_b, mask_b)

    # 3. 画像(image)の枚数を増やす
    # 1枚しかなくて、他が複数枚あるなら、target_bまでコピーして増やす
    if img_b == 1 and target_b > 1:
        image = image.expand(target_b, -1, -1, -1).clone()

    # 4. マスク(mask)の枚数を増やす
    if mask_b == 1 and target_b > 1:
        mask = mask.expand(target_b, -1, -1).clone()

    # 5. 最終チェック（整合性が取れているか）
    # 枚数が 1 でも target_b でもない中途半端な数（例: 3枚と5枚など）ならエラー
    if not (image.shape[0] == mask.shape[0]):
        raise ValueError(f"Batch sizes are incompatible: "
                         f"Image({image.shape[0]}), Mask({mask.shape[0]})")

    return image, mask
    

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_fill_holes
import comfy.utils



def fill_mask_holes(mask: torch.Tensor):
    """
    複数の閾値を用いて、マスクの穴を段階的に埋める
    """
    # 1. 形状の準備 [B, H, W] -> [B, 1, H, W]
    was_3d = mask.dim() == 3
    if was_3d:
        mask = mask.unsqueeze(1)
    
    device = mask.device
    dtype = mask.dtype
    thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # 2. しきい値ごとにループ（ここは論理上必要）
    for threshold in thresholds:
        # A. しきい値で二値化
        binary_mask = (mask >= threshold).float()
        
        # B. Closing処理 (3x3の矩形構造要素) をPyTorchで高速実行
        # 膨張 (max_pool) して 収縮 (min_pool)
        # ※ binary_closing(structure=ones(3,3)) と同等
        closed = F.max_pool2d(binary_mask, kernel_size=3, stride=1, padding=1)
        closed = -F.max_pool2d(-closed, kernel_size=3, stride=1, padding=1) # min_poolの代わり
        
        # C. 穴埋め処理 (Scipyを使用)
        # ここだけはScipyが非常に強力なため、バッチ単位でNumPyに変換
        closed_np = closed.squeeze(1).cpu().numpy() > 0.5
        filled_np = np.zeros_like(closed_np)
        
        for b in range(closed_np.shape[0]):
            filled_np[b] = binary_fill_holes(closed_np[b])
            
        filled_torch = torch.from_numpy(filled_np).to(device=device, dtype=dtype).unsqueeze(1)
        
        # D. 元のサンプルを更新 (filled部分にthresholdの値を書き込む)
        # mask_np = np.maximum(mask_np, np.where(filled_mask != 0, threshold, 0)) と同等
        update_val = filled_torch * threshold
        mask = torch.max(mask, update_val)

    # 3. 形状を戻す
    if was_3d:
        mask = mask.squeeze(1)
        
    return mask


def filter_mask_high_pass(mask: torch.Tensor, threshold: float=0.1):
    filtered_mask = mask.clone()
    filtered_mask[filtered_mask < threshold] = 0
    return filtered_mask


def expand_mask(mask: torch.Tensor, pixels: int):
    kernel_size = pixels * 2 + 1
    mask_in = mask.unsqueeze(1) # [B, H, W] -> [B, 1, H, W]
    dilated = F.max_pool2d(mask_in, kernel_size=kernel_size, stride=1, padding=pixels)
    return dilated.squeeze(1)


def blur_mask(mask: torch.Tensor, pixels: int):
    sigma = pixels / 4
    kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
    
    x = torch.arange(kernel_size, device=mask.device, dtype=mask.dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size)

    mask_in = mask.unsqueeze(1)
    pad = kernel_size // 2

    mask_padded = F.pad(mask_in, (pad, pad, pad, pad), mode="replicate")
    blurred = F.conv2d(mask_padded, kernel_2d, padding=0, groups=1)

    return blurred.squeeze(1).clamp(0.0, 1.0)


def find_context_area(mask: torch.Tensor):
    """
    マスクが存在する範囲(Bounding Box)を特定する。
    戻り値: x_min, y_min, width, height (すべて [B] 形状のテンソル)
    """
    B, H, W = mask.shape
    device = mask.device

    # 1. 各行・各列にマスク（値 > 0）が存在するかを判定
    # [B, H, W] -> [B, H] / [B, W]
    any_y = mask.any(dim=2) 
    any_x = mask.any(dim=1)

    def get_min_max(any_dim, size):
        # 0 から size-1 までのインデックス
        indices = torch.arange(size, device=device) # [size]

        # マスクがある場所はそのインデックス、ない場所は min用にsize / max用に-1 を置く
        # ブロードキャストにより [B, size] と [size] が自動的に適合する
        min_idx = torch.where(any_dim, indices, size).min(dim=1).values
        max_idx = torch.where(any_dim, indices, -1).max(dim=1).values

        # そのバッチに全くマスクがない（すべてFalse）場合の処理
        has_mask = any_dim.any(dim=1)
        min_idx[~has_mask] = -1
        max_idx[~has_mask] = -1

        return min_idx, max_idx

    # Y（高さ方向）と X（幅方向）の最小・最大インデックスを取得
    y_min, y_max = get_min_max(any_y, H)
    x_min, x_max = get_min_max(any_x, W)

    # 2. 幅(w)と高さ(h)の計算
    # マスクが存在する場合のみ (max - min + 1) を計算し、存在しない場合は -1
    w = torch.where(x_min >= 0, x_max - x_min + 1, -1)
    h = torch.where(y_min >= 0, y_max - y_min + 1, -1)
        
    return x_min, y_min, w, h


def expand_context_area(mask: torch.Tensor, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, h: torch.Tensor, extend_factor: float):
    img_h, img_w = mask.shape[1], mask.shape[2]
    device = mask.device

    grow_x = (w.float() * (extend_factor - 1.0) / 2.0).round().long()
    grow_y = (h.float() * (extend_factor - 1.0) / 2.0).round().long()

    new_x = torch.clamp(x - grow_x, min=0)
    new_y = torch.clamp(y - grow_y, min=0)
    new_x2 = torch.clamp(x + w + grow_x, max=img_w)
    new_y2 = torch.clamp(y + h + grow_y, max=img_h)
    
    new_w = new_x2 - new_x
    new_h = new_y2 - new_y
    
    empty = (w == -1)
    new_x[empty] = 0
    new_y[empty] = 0
    new_w[empty] = img_w
    new_h[empty] = img_h
    
    return new_x, new_y, new_w, new_h


def pad_to_multiple(val: int, p: int):
    """
    値を指定された倍数に切り上げる
    """
    if p <= 0: return val
    return (val + p - 1) // p * p


def calculate_crop_region(x: int, y: int, w: int, h: int, target_w: int, target_h: int, img_w: int, img_h: int):
    """
    ターゲットのアスペクト比に合わせて、最適なクロップ範囲(x, y, w, h)を計算する
    """
    target_aspect = target_w / target_h
    current_aspect = w / h

    # アスペクト比を合わせるために範囲を広げる
    if current_aspect < target_aspect:
        new_w = int(h * target_aspect)
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y
    else:
        new_w = w
        new_h = int(w / target_aspect)
        new_x = x
        new_y = y - (new_h - h) // 2
    
    # はみ出し位置の調整(できるだけ画像内に収める)
    def adjust_axis(pos: int, size: int, limit: int):
        if pos < 0:
            # 左/上 がはみ出している場合、右/下 に余裕があればずらす
            shift = -pos
            if pos + size + shift <= limit: 
                return pos + shift
            return -((size - limit) // 2)
        
        elif pos + size > limit:
            # 右/下 がはみ出している場合、左/上 に余裕があればずらす
            overflow = pos + size - limit
            if pos - overflow >= 0:
                return pos - overflow
            return -((size - limit) // 2)
        
        return pos
    
    new_x = adjust_axis(new_x, new_w, img_w)
    new_y = adjust_axis(new_y, new_h, img_h)
    return new_x, new_y, new_w, new_h


def create_padded_canvas(image: torch.Tensor, mask: torch.Tensor, new_x: int, new_y: int, new_w: int, new_h: int):
    """
    画像外をエッジピクセルで埋めたキャンバスを作成し、座標情報を返す
    """
    B, H, W, C = image.shape
    
    # パディング量の計算
    up = max(0, -new_y)
    down = max(0, (new_y + new_h) - H)
    left = max(0, -new_x)
    right = max(0, (new_x + new_w) - W)

    # 画像のパディング(エッジリピート)
    img_t = image.permute(0, 3, 1, 2)
    img_t = F.pad(img_t, (left, right, up, down), mode="replicate")
    canvas_image = img_t.permute(0, 2, 3, 1)

    # マスクのパディング
    mask_t = mask.unsqueeze(1)
    mask_t = F.pad(mask_t, (left, right, up, down), mode="constant", value=1.0)
    canvas_mask = mask_t.squeeze(1)

    # 元画像がキャンバス内のどこにあるかの座標
    canvas_to_original = (left, up, W, H)
    return canvas_image, canvas_mask, canvas_to_original


def resize_image(image: torch.Tensor, size: tuple, algorithm: str):
    dim = image.dim()
    if dim == 3:
        image = image.unsqueeze(0)
    
    image = image.movedim(-1, 1) # [B, C, H, W]
    if algorithm == "lanczos":
        height, width = size
        res = comfy.utils.lanczos(image, int(width), int(height))
    else:
        res = F.interpolate(image, size=size, mode=algorithm)
    
    res = res.movedim(1, -1) # [B, H, W, C]
    
    if dim == 3:
        res = res.squeeze(0)
    return res


def resize_mask(mask: torch.Tensor, size: tuple, algorithm: str):
    dim = mask.dim()
    if dim == 2:
        mask = mask.unsqueeze(0)
    elif dim != 3:
        raise ValueError(f"Expected mask tensor [B, H, W] or [H, W], got {dim}D (shape: {mask.shape})")
    
    res = resize_image(mask.unsqueeze(-1), size, algorithm).squeeze(-1).clamp(0.0, 1.0)

    if dim == 2:
        res = res.squeeze(0)
    return res
    



def advanced_crop(
    image: torch.Tensor, 
    mask: torch.Tensor, 
    x: int, y: int, w: int, h: int, 
    target_w: int, target_h: int, padding: int, 
    algorithm: str,
    resize_output: bool = True,
):
    """
    メインのクロップ関数
    """
    # 安全性チェック
    if target_w <= 0 or target_h <= 0 or w <= 0 or h <= 0:
        return image, mask, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]
    
    # クロップ領域の計算
    img_h, img_w = image.shape[1], image.shape[2]
    if resize_output:
        # ターゲット解像度の調整
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)
        new_x, new_y, new_w, new_h = calculate_crop_region(x, y, w, h, target_w, target_h, img_w, img_h)
    else:
        new_x, new_y, new_w, new_h = x, y, w, h

    # キャンバスの拡張
    canvas_image, canvas_mask, (cto_x, cto_y, cto_w, cto_h) = create_padded_canvas(image, mask, new_x, new_y, new_w, new_h)

    # クロップ実行
    ctc_x = new_x + cto_x
    ctc_y = new_y + cto_y
    ctc_w = new_w
    ctc_h = new_h
    cropped_image = canvas_image[:, ctc_y:ctc_y + new_h, ctc_x:ctc_x + new_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + new_h, ctc_x:ctc_x + new_w]
    
    # リサイズ
    if resize_output:
        target_size = (target_h, target_w)
        cropped_image = resize_image(cropped_image, target_size, algorithm)
        cropped_mask = resize_mask(cropped_mask, target_size, algorithm)
    
    
    return canvas_image, canvas_mask, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h
    

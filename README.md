# comfy-jupo-detailer

ComfyUI用のDetailer補助ノード集です。

マスク領域の切り出し、内部KSamplerによるinpaint/detail処理、元画像への合成、アウトペイント用キャンバスの拡張をまとめて扱えます。

## Installation

`custom_nodes` に配置して、ComfyUIを再起動してください。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jupo-ai/comfy-jupo-detailer.git
```

この拡張の `requirements.txt` は空です。追加の依存関係をインストールする必要はありません。

## Nodes

### Detailer Cropper

`images` と `masks` を受け取り、マスク領域を元に処理用画像とマスクを切り出します。

Outputs:

- `crop_info`: `Detailer` / `Detailer Stitcher` へ渡すクロップ情報
- `cropped_images`: 切り出した画像
- `cropped_masks`: 切り出したマスク

主な入力:

- `mask_fill_holes`: マスクの小さな穴を埋めます。
- `mask_expand_pixels`: マスクを指定ピクセル分拡張します。
- `mask_blend_pixels`: 合成境界用にマスクを拡張し、ぼかします。
- `mask_filter_threshold`: 指定値より小さいマスク値を除去します。
- `context_extend_factor`: マスク領域からクロップ範囲を広げる倍率です。
- `output_resize`: クロップ後の出力サイズを制御します。
- `resize_algorithm`: リサイズ方式です。
- `output_padding`: 幅と高さがこの倍数になるように調整します。

`output_resize`:

- `None`: リサイズしません。複数マスクの場合のみ、2枚目以降を1枚目のサイズに合わせます。
- `keep aspect`: アスペクト比を保ちつつ、面積が `output_resolution ** 2` に近くなるようにします。
- `constant`: `output_width` / `output_height` に合わせます。

### Detailer Outpainter

画像を上下左右へ拡張し、拡張部分をマスクとして返します。`Detailer` へ接続するためのアウトペイント用ノードです。

Outputs:

- `crop_info`: `Detailer` / `Detailer Stitcher` へ渡すクロップ情報
- `outpaint_images`: 拡張後の画像
- `outpaint_masks`: 拡張部分のマスク

主な入力:

- `extend_up_pixels` / `extend_down_pixels`: 上下方向の拡張量です。
- `extend_left_pixels` / `extend_right_pixels`: 左右方向の拡張量です。
- `mask_blend_pixels`: 拡張部分のマスク境界をぼかします。
- `output_resize`: `Detailer` へ渡す画像サイズを制御します。
- `resize_algorithm`: リサイズ方式です。
- `output_padding`: 幅と高さがこの倍数になるように調整します。

拡張部分の画像は、元画像の端を引き伸ばして作られます。マスクは拡張部分が `1.0`、元画像部分が `0.0` です。

### Detailer

`crop_info`、Cropper/Outpainterの画像とマスクを受け取り、内部でKSamplerを実行します。

Inputs:

- `model`
- `vae`
- `positive`
- `negative`
- `crop_info`
- `cropped_images`
- `cropped_masks`
- `seed`
- `steps`
- `cfg`
- `sampler_name`
- `scheduler`
- `denoise`
- `noise_mask`

Outputs:

- `crop_info`: 入力されたクロップ情報
- `inpainted`: KSampler処理後の画像

`positive` / `negative` は通常の `CONDITIONING` に加えて、crop数と同じ長さのconditioningリストも扱えます。

`noise_mask=True` の場合、KSamplerのノイズ適用範囲がマスク領域に制限されます。`False` の場合はcrop全体がimg2img寄りに変化します。

### Detailer Stitcher

`Detailer` などで処理した画像を、`crop_info` に従って元画像へ戻します。

Inputs:

- `crop_info`
- `inpainted`
- `override`

Outputs:

- `images`: 合成後の画像

`override=True` の場合、同じ元画像から複数cropされた結果を1枚の画像へ順番に戻します。たとえば1枚の画像に2つのマスクがある場合、出力も1枚になります。

`override=False` の場合、cropごとに1枚ずつ戻した画像を返します。

## Basic Workflows

### Inpaint / Detail

```text
IMAGE + MASK
  -> Detailer Cropper
  -> Detailer
  -> Detailer Stitcher
  -> IMAGE
```

### Outpaint

```text
IMAGE
  -> Detailer Outpainter
  -> Detailer
  -> Detailer Stitcher
  -> expanded IMAGE
```

## Notes

- `crop_info` は `Detailer Cropper` / `Detailer Outpainter` から `Detailer` / `Detailer Stitcher` へそのまま渡してください。
- `cropped_images` / `cropped_masks` と `outpaint_images` / `outpaint_masks` は、どちらも `Detailer` の画像・マスク入力へ接続できます。
- `Detailer Cropper` と `Detailer Outpainter` の `output_resize` に応じて、不要なサイズ入力はフロントエンド側で自動的に非表示になります。

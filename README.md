# comfy-jupo-detailer

ComfyUI用のDetailer補助ノード集です。

マスクから処理対象を切り出す `Detailer Cropper`、切り出した画像を内部KSamplerで処理する `Detailer`、元画像へ戻す `Detailer Stitcher`、アウトペイント用に画像を拡張する `Detailer Outpainter`、画像ごとにタグを推定する `Tagger` を含みます。

## Installation

`custom_nodes` に配置し、依存関係をインストールしてください。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jupo-ai/comfy-jupo-detailer.git
cd ../
python -m pip install -r custom_nodes/comfy-jupo-detailer/requirements.txt
```

ComfyUI本体の `requirements.txt` に含まれる依存は、この拡張側では重複していません。追加で必要なのは主にTagger用の `onnxruntime` です。

## Nodes

### Detailer Cropper

`images` と `masks` を受け取り、マスク領域を元に処理用画像を切り出します。

Outputs:

- `crop_info`: Stitcherで元画像へ戻すための情報
- `cropped_images`: 切り出した画像
- `cropped_masks`: 切り出したマスク

主な入力:

- `mask_fill_holes`: マスクの穴埋め
- `mask_expand_pixels`: マスク拡張
- `mask_blend_pixels`: 合成境界用のマスクぼかし
- `mask_filter_threshold`: 小さいマスク値の除去
- `context_extend_factor`: マスク領域からクロップ範囲を広げる倍率
- `output_resize`: 出力サイズ制御

`output_resize`:

- `None`: リサイズしません。複数マスクの場合のみ、`torch.stack` のため2枚目以降を1枚目のサイズに合わせます。
- `keep aspect`: アスペクト比を保ちつつ、面積が `output_resolution ** 2` に近くなるようにします。
- `constant`: `output_width` / `output_height` に合わせます。

### Detailer Outpainter

画像を上下左右へ拡張し、拡張部分をマスクとして返します。Detailerへ接続するためのアウトペイント用ノードです。

Outputs:

- `crop_info`
- `outpaint_images`
- `outpaint_masks`

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
- KSampler設定一式

Outputs:

- `crop_info`
- `inpainted`

`positive` / `negative` は通常の `CONDITIONING` だけでなく、crop数と同じ長さのconditioningリストも想定しています。Taggerで画像ごとにタグ付けし、標準の `CLIP Text Encode` でconditioning化して接続する用途を想定しています。

`noise_mask=True` の場合、KSamplerのノイズ適用範囲がマスク領域に制限されます。`False` の場合はcrop全体がimg2img寄りに変化します。

### Detailer Stitcher

`Detailer` などで処理した画像を `crop_info` に従って戻します。

Inputs:

- `crop_info`
- `inpainted`
- `override`

`override=True` の場合、同じ元画像から複数cropされた結果を1枚の画像へ順番に戻します。たとえば1枚の画像に2つのマスクがある場合、出力も1枚になります。

`override=False` の場合、cropごとに1枚ずつ戻した画像を返します。

### Tagger

画像ごとにWD系taggerモデルでタグを推定し、`list[str]` として返します。

対応モデル:

- `wd-eva02-large-tagger-v3`
- `wd-vit-large-tagger-v3`
- `wd-v1-4-swinv2-tagger-v2`
- `wd-vit-tagger-v3`

Inputs:

- `images`
- `model`
- `threshold`
- `character_threshold`
- `replace_underscore`
- `trailing_comma`
- `exclude_tags`

Taggerモデルは初回実行時に自動ダウンロードされます。

保存先は、ComfyUIの `extra_model_paths.yaml` で `tagger` が登録されている場合はそのディレクトリを使います。登録されていない場合は、この拡張直下の `tagger_models` を使用します。

例:

```yaml
my_models:
  base_path: /path/to/models
  tagger: tagger
```

## Basic Workflows

### Inpaint / Detail

```text
IMAGE + MASK
  -> Detailer Cropper
  -> Detailer
  -> Detailer Stitcher
  -> IMAGE
```

cropごとにプロンプトを変えたい場合:

```text
Detailer Cropper.cropped_images
  -> Tagger
  -> CLIP Text Encode
  -> Detailer.positive
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

- `Detailer Cropper` と `Detailer Outpainter` の `output_resize` に応じて、不要なサイズ入力はフロントエンド側で自動的に非表示になります。
- `crop_info` は `Detailer Cropper` / `Detailer Outpainter` から `Detailer` / `Detailer Stitcher` へそのまま渡してください。
- Taggerモデルのダウンロードにはインターネット接続が必要です。

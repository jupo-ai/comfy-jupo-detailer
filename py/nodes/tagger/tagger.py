from comfy_api.latest import io
from .common import PACKAGE_NAME, CATEGORY
from ...utils import mk_name

import comfy.utils
import torch

from . import model_manager
from ...modules import tagger as tagger_module


class Tagger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "Tagger"),
            display_name="Tagger",
            category=CATEGORY,
            inputs=[
                io.Image.Input("images", tooltip="タグ付けする画像。複数画像の場合は各画像ごとにタグ文字列を出力する。"),
                io.Combo.Input("model", options=model_manager.KNOWN_TAGGERS, tooltip="使用するTaggerモデル。未保存の場合は初回実行時に自動ダウンロードする。"),
                io.Float.Input("threshold", default=0.35, min=0.0, max=1.0, step=0.01, tooltip="generalタグを採用する信頼度のしきい値。値を上げるほどタグ数が少なくなる。"),
                io.Float.Input("character_threshold", default=0.85, min=0.0, max=1.0, step=0.01, tooltip="characterタグを採用する信頼度のしきい値。キャラクター名などのタグに適用する。"),
                io.Boolean.Input("replace_underscore", default=True, tooltip="タグ内のアンダースコアをスペースに置換する。例: red_hair -> red hair"),
                io.Boolean.Input("drop_overlap", default=True, tooltip="重複・包含関係にあるタグを削除する。例: very_long_hair がある場合に long_hair を削除する。"),
                io.Boolean.Input("drop_blacklist", default=True, tooltip="imgutils互換のblacklistプリセットを使い、不要になりやすいタグを削除する。初回使用時にフィルタ用データを取得する。"),
                io.Boolean.Input("drop_basic_character", default=False, tooltip="red_hair や blue_eyes など、キャラクター外見の基本属性タグを削除する。"),
                io.Combo.Input("sort_mode", options=["original", "score", "shuffle"], tooltip="タグの並び順。original: モデル出力順、score: 信頼度順、shuffle: 人数タグ以外をランダム化。"),
                io.Boolean.Input("prioritize_people_tags", default=True, tooltip="solo や 1girl などの人数タグを先頭に寄せる。無効にすると sort_mode の結果をそのまま使う。"),
                io.Boolean.Input("trailing_comma", default=False, tooltip="出力文字列の末尾にもカンマとスペースを付ける。"),
                io.String.Input("exclude_tags", default="", tooltip="出力から除外するタグをカンマ区切りで指定する。例: watermark, signature"),
            ],
            outputs=[
                io.String.Output(is_output_list=True),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        model: str,
        threshold: float = 0.35,
        character_threshold: float = 0.85,
        replace_underscore: bool = True,
        drop_overlap: bool = True,
        drop_blacklist: bool = True,
        drop_basic_character: bool = False,
        sort_mode: str = "original",
        prioritize_people_tags: bool = True,
        trailing_comma: bool = False,
        exclude_tags: str = "",
    ):
        manager = model_manager.ModelManager()
        download_pbar = comfy.utils.ProgressBar(1)
        if not manager.download_model(model, download_pbar):
            raise RuntimeError(f"Failed to prepare tagger model: {model}")
        download_pbar.update_absolute(1, 1)

        inference_pbar = comfy.utils.ProgressBar(images.shape[0])
        results = tagger_module.tag_images(
            images=images,
            model=model,
            model_paths=manager.get_model_paths(model),
            threshold=threshold,
            character_threshold=character_threshold,
            replace_underscore=replace_underscore,
            trailing_comma=trailing_comma,
            exclude_tags=exclude_tags,
            drop_overlap=drop_overlap,
            drop_blacklist=drop_blacklist,
            drop_basic_character=drop_basic_character,
            sort_mode=sort_mode,
            prioritize_people_tags=prioritize_people_tags,
            pbar=inference_pbar,
        )

        return io.NodeOutput(results)

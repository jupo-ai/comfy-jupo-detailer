from pathlib import Path
import folder_paths
import requests
from tqdm import tqdm

from ...modules.tagger import KNOWN_TAGGERS, get_model_spec

DEFAULT_MODELS_DIR = Path(__file__).parents[3] / "tagger_models"


def get_models_dir() -> Path:
    if "tagger" in folder_paths.folder_names_and_paths:
        paths = folder_paths.get_folder_paths("tagger")
        if paths:
            return Path(paths[0])
    return DEFAULT_MODELS_DIR


class ModelManager:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else get_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    
    def get_model_path(self, model_name: str, ext: str="onnx") -> Path:
        if ext.startswith("."):
            ext = ext.lstrip(".")
        return self.models_dir / f"{model_name}.{ext}"


    def get_model_paths(self, model_name: str) -> dict[str, Path]:
        spec = get_model_spec(model_name)
        return {
            file.filename: self.get_model_path(model_name, file.ext)
            for file in spec.files
            if self.get_model_path(model_name, file.ext).exists() or file.required
        }
    
    
    def download_model(self, model_name: str, progress_bar=None) -> bool:
        spec = get_model_spec(model_name)
        base_url = f"https://huggingface.co/{spec.repo_id}/resolve/main"

        download_jobs = []
        for file in spec.files:
            dest = self.get_model_path(model_name, file.ext)
            if not dest.exists():
                download_jobs.append((f"{base_url}/{file.filename}", dest, file.required))

        progress_state = None
        if progress_bar is not None and download_jobs:
            progress_state = {
                "downloaded": 0,
                "total": sum(self._get_content_length(url) for url, _, _ in download_jobs),
            }

        for url, dest, required in download_jobs:
            if not self._download_file(url, dest, progress_bar, progress_state):
                if required:
                    return False

        return True
    
    
    def _get_content_length(self, url: str) -> int:
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            response.raise_for_status()
            return int(response.headers.get("content-length", 0))
        except Exception:
            return 0


    def _download_file(self, url: str, dest_path: Path, progress_bar=None, progress_state=None) -> bool:
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {dest_path.name}",
            ) as pbar:
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            chunk_size = len(chunk)
                            downloaded += chunk_size
                            pbar.update(chunk_size)
                            if progress_bar is not None:
                                if progress_state is not None and progress_state["total"] > 0:
                                    progress_state["downloaded"] += chunk_size
                                    current = min(progress_state["total"], progress_state["downloaded"])
                                    progress_bar.update_absolute(current, progress_state["total"])
                                elif total_size > 0:
                                    current = min(total_size, downloaded)
                                    progress_bar.update_absolute(current, total_size)
            
            return True
        
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            if dest_path.exists():
                dest_path.unlink() # 不完全なファイルを削除
            return False

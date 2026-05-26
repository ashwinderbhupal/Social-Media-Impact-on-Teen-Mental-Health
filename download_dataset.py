from pathlib import Path
import shutil

import kagglehub

DATASET_SLUG = "algozee/teenager-menthal-healy"
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"


def download_dataset() -> tuple[Path, Path]:
    """Download from Kaggle and copy CSV into project data/."""
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(cache_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cache_path}")

    dest = PROJECT_DATA_DIR / csv_files[0].name
    shutil.copy2(csv_files[0], dest)
    return dest, cache_path


if __name__ == "__main__":
    project_path, cache_path = download_dataset()
    print("Path to dataset files (project):", project_path)
    print("Path to dataset files (cache):", cache_path)

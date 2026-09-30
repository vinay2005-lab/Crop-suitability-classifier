from pathlib import Path
import urllib.request

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "Crop_recommendation.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/the-amazing-atharva/"
    "Crop-Recommendation/main/Crop_recommendation.csv"
)

def download_dataset(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists() and not force:
        return DATA_PATH

    print("Downloading Crop Recommendation dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Check your internet connection "
            "or manually place Crop_recommendation.csv inside the data/ folder."
        ) from exc

    print(f"Dataset saved to: {DATA_PATH}")
    return DATA_PATH

if __name__ == "__main__":
    download_dataset()

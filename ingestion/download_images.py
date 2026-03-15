import json
import requests
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/nature_metadata.json")
IMAGE_DIR = Path("data/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def download_image(url: str, save_path: Path):
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    else:
        return False


def main():
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total images in metadata: {len(data)}")

    for idx, item in enumerate(data):
        image_url = item["urls"]["regular"]
        image_id = item["id"]

        image_path = IMAGE_DIR / f"{image_id}.jpg"

        if image_path.exists():
            continue

        success = download_image(image_url, image_path)

        if success:
            print(f"[{idx+1}] Downloaded {image_id}")
        else:
            print(f"[{idx+1}] Failed {image_id}")


if __name__ == "__main__":
    main()

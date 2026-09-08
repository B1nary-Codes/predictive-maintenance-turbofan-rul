import os
import zipfile
import urlib.request
from pathlib import path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_URL = "https://data.nasa.gov/api/views/1922-2322/rows.zip?accessType=DOWNLOAD"

def download_and_extract():
    """Downloads NASA C-MAPSS dataset and extracts text files to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "cmapss.zip"

    if not any(RAW_DIR.glob("*.txt")):
        print(f"Downloading NASA C-MAPSS dataset to {RAW_DIR}...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
            
        if zip_path.exists():
            os.remove(zip_path)
        print("Dataset ready in data/raw/")
    else:
        print("Raw data files already present in data/raw/")

if __name__ == "__main__":
    download_and_extract()

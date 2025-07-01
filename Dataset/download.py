import os
import urllib.request
import argparse
from zipfile import ZipFile

# Dataset URL
url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
filename = "dataset.zip"
extract_folder = "FUNSD"

def download_funsd():
    # Create directory to store the dataset
    os.makedirs(extract_folder, exist_ok=True)

    # Download the zip file
    if not os.path.exists(filename):
        print(f"Downloading FUNSD dataset...")
        urllib.request.urlretrieve(url, filename)
        print("\nDownload complete.")
    else:
        print("dataset.zip already exists. Skipping download.")

    # Extract the contents
    print("Extracting dataset...")
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    print("✅ FUNSD dataset is ready in the 'FUNSD/' folder.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and extract datasets for DocKylin.")
    parser.add_argument("--dataset", type=str, default="funsd", help="Name of the dataset to download (e.g., 'funsd').")
    args = parser.parse_args()

    if args.dataset.lower() == "funsd":
        download_funsd()
    else:
        print(f"Unknown dataset: {args.dataset}. Currently only 'funsd' is supported.")

import os
import zipfile
import sys
from tqdm import tqdm


def setup_kaggle():
    """Setup Kaggle API with proper error handling"""
    try:
        os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
        import kaggle

        kaggle_dir = os.getcwd() 
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_json):
            print("Error: kaggle.json not found in current directory")
            print("Please follow these steps:")
            print("1. Go to https://www.kaggle.com/")
            print("2. Sign in and go to Settings → API")
            print("3. Click 'Create New API Token'")
            print("4. Move the downloaded kaggle.json to this directory:")
            print(f"   {os.getcwd()}")
            print("5. Run: chmod 600 kaggle.json")
            return False
        
        print("✅ Kaggle API credentials found in current directory")
        return True
        
    except ImportError:
        print("Kaggle package not installed")
        print("Install it with: pip install kaggle")
        return False


def download_covid_dataset(dataset="tawsifurrahman/covid19-radiography-database", 
                           output_path="./data/raw"):
    """
    Download the COVID-19 Radiography dataset from Kaggle if not already present.
    """
   
    os.makedirs(output_path, exist_ok=True)

   
    if os.listdir(output_path):
        print(f"Dataset already exists in {output_path}, skipping download.")
        return True

  
    if not setup_kaggle():
        print("Cannot download dataset - Kaggle API not configured")
        return False
    
    try:
        # Set environment variable again before using kaggle
        os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
        import kaggle
        
        print(f"Downloading dataset: {dataset}")
        kaggle.api.dataset_download_files(dataset, path=output_path, unzip=True)
        print(f"Dataset downloaded and extracted to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


if __name__ == "__main__":
    success = download_covid_dataset()
    if success:
        print("Dataset ready!")
    else:
        print("Dataset download failed. Please check the setup instructions above.")
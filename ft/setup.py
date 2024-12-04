
import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files('mexwell/stress-detection-from-social-media-articles', path='.local/stress/raw', unzip=True)
from dotenv import load_dotenv
import kaggle

load_dotenv("/Users/marcalph/.ssh/llm_api_keys.env")
kaggle.api.authenticate()
kaggle.api.dataset_download_files('mexwell/stress-detection-from-social-media-articles', path='.local/stress-detection-from-social-media-articles', unzip=True)
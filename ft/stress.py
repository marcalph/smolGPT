import pandas as pd
from openai import OpenAI
import os
import json
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from pprint import pprint

load_dotenv("/Users/marcalph/.ssh/llm_api_keys.env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_stress_data(raw_path:Path) -> pd.DataFrame:
  data = pd.read_csv(raw_path/"Reddit_Title.csv", sep=';')
  data_cleaned = data[['title', 'label']].head(5000)
  label_mapping = {0: "non-stress", 1: "stress"}
  data_cleaned['label'] = data_cleaned['label'].map(label_mapping)
  return data_cleaned


def df_to_jsonl(data:pd.DataFrame, output_file_path:Path) -> None:
  jsonl_data = []
  for _, row in data.iterrows():
    jsonl_data.append({
      "messages": [
        {"role": "system", "content": "Given a social media post, classify whether it indicates 'stress' or 'non-stress'."},
        {"role": "user", "content": row['title']},
        {"role": "assistant", "content": f"\"{row['label']}\""}
      ]
    })

  with open(output_file_path, 'w') as f:
    for item in jsonl_data:
      f.write(json.dumps(item) + '\n')
  logger.info(f"Saved {len(jsonl_data)} items to {output_file_path.name}")


def split_and_serialize_to_jsonl(df:pd.DataFrame, output_dir:Path, fragment_trainset:bool=True) -> None:
  # split data into training and validation
  train_data, validation_data = train_test_split(df, test_size=0.2, random_state=42)
  
  # serialize data
  os.makedirs(output_dir, exist_ok=True)

  if fragment_trainset:
    for i in np.linspace(len(train_data)//10, len(train_data), num=10):
      train_output_file_path = output_dir/f'stress_detection_train_{int(i)}.jsonl'
      df_to_jsonl(train_data.iloc[:int(i)], train_output_file_path)
  else:
    train_output_file_path = output_dir/'stress_detection_train.jsonl'
    df_to_jsonl(train_data, train_output_file_path)

  validation_output_file_path = output_dir/'stress_detection_validation.jsonl'
  df_to_jsonl(validation_data, validation_output_file_path)
  logger.info(f"jsonl files saved to {output_dir}")



def upload_datasets(dir_path:Path, client:OpenAI) -> None:
  stress_datasets_mapping = {}
  for file_path in dir_path.glob("*.jsonl"):
    fileobj = client.files.create(
      file=open(file_path, 'rb'), 
      purpose="fine-tune")
    logger.info(f"Uploaded {file_path.name} to OpenAI")
    stress_datasets_mapping[file_path.name] = fileobj.id
  # save mapping to file
  with open(dir_path.parent/"utils/stress_datasets_mapping.json", 'w') as json_file:
    json.dump(stress_datasets_mapping, json_file)
  

if __name__ == "__main__":
  from utils import STRESS_RAW_PATH
  # load data
  data_cleaned = load_stress_data(STRESS_RAW_PATH)
  output_dir = STRESS_RAW_PATH.parent/"processed"
  split_and_serialize_to_jsonl(data_cleaned, output_dir, fragment_trainset=True)
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MLEXPERIMENTS"))
  upload_datasets(output_dir, client)

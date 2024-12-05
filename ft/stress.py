import pandas as pd
from openai import OpenAI
import os
import json
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from wandb.integration.openai.fine_tuning import WandbLogger

load_dotenv("/Users/marcalph/.ssh/llm_api_keys.env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)

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


def split_and_serialize_to_jsonl(df:pd.DataFrame, output_dir:Path, serialize:bool=True, fragment_trainset:bool=True) -> None:
  # split data into training and validation
  train_data, validation_data = train_test_split(df, test_size=0.2, random_state=42)
  
  if not serialize:
    return train_data, validation_data
    
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
  return train_data, validation_data



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


def finetune_test(openai_client:OpenAI, model="gpt-3.5-turbo") -> str:
  ftmodel = openai_client.fine_tuning.jobs.create(
    model=model,
    training_file="file-Rzq4uHT6uenTsqLBDMk7b2", # train 400
    validation_file="file-JD8qkemtmDCdgH9xs9i5ce", # validation
    hyperparameters={
      "n_epochs": 3,
      "batch_size": 3,
      "learning_rate_multiplier": 2
  }
  )
  ftjob_id = ftmodel.id
  status = ftmodel.status

  logger.info(f'Fine-tuning model with jobID: {ftjob_id}.')
  logger.info(f"Training Response: {ftmodel}")
  logger.info(f"Training Status: {status}")
  # todo check need for  of kwargs_wandb_init.config
  WandbLogger.sync(
    fine_tune_job_id=ftjob_id, 
    openai_client=openai_client,
    project="ml-experiments", 
    wait_for_job_success=False,
    tags=["ft", "openai", "stress", "400"])
  return ftjob_id


def predict(openai_client: OpenAI, test_df: pd.DataFrame,  model:str):
    y_pred = []
    categories = ["non-stress", "stress"]
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Processing rows"):
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Given a social media post, classify whether it indicates 'stress' or 'non-stress'.",
                },
                {"role": "user", "content": row["title"]},
            ],
        )

        answer = response.choices[0].message.content

        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("none")
    return y_pred



def run_evaluation(y_true: np.ndarray, y_pred:np.ndarray):
    labels = ["non-stress", "stress"]
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(
            x, -1
        )  # Map to -1 if not found, but should not occur with correct data

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Calculate accuracy

    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f"Accuracy: {accuracy:.3f}")

    # Generate accuracy report

    unique_labels = set(y_true_mapped)  # Get unique labels

    for label in unique_labels:
        label_indices = [
            i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label
        ]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f"Accuracy for label {labels[label]}: {label_accuracy:.3f}")
    # Generate classification report

    class_report = classification_report(
        y_true=y_true_mapped,
        y_pred=y_pred_mapped,
        target_names=labels,
        labels=list(range(len(labels))),
    )
    print("\nClassification Report:")
    print(class_report)

    conf_matrix = confusion_matrix(
        y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels)))
    )
    print("\nConfusion Matrix:")
    print(conf_matrix)





if __name__ == "__main__":
  from utils import STRESS_RAW_PATH
  output_dir = STRESS_RAW_PATH.parent/"processed"
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MLEXPERIMENTS"))
  prepare_data = False
  train = True
  evaluate = False

  if prepare_data:
    # load data
    data_cleaned = load_stress_data(STRESS_RAW_PATH)
    # process data
    split_and_serialize_to_jsonl(data_cleaned, output_dir, serialize=True, fragment_trainset=True)
    # upload
    upload_datasets(output_dir, client)
  if train:
    for _ in range(5):
      finetune_test(client)
  if evaluate:
    data_cleaned = load_stress_data(STRESS_RAW_PATH)
    _, test_df = split_and_serialize_to_jsonl(data_cleaned, output_dir, serialize=False)
    y_pred = predict(client, test_df, model="gpt-3.5-turbo")  
    run_evaluation(test_df['label'], y_pred)
    y_pred = predict(client, test_df, model="ft:gpt-3.5-turbo-0125:wingmate::AarlN1qI")  
    run_evaluation(test_df['label'], y_pred)
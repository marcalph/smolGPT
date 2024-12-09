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
import wandb
import glob
import tempfile
from utils import STRESS_RAW_PATH, WANDB_PROJECT
from enum import Enum


class ArtifactType(Enum):
    RAW = "raw"
    PREP = "preprocessed"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    COMPLETION = "completion"
    CONVERSATION_CHUNK = "conversation_chunk"


class WandbJobType(Enum):
    UPLOAD = "upload"
    SPLIT = "split"
    FINE_TUNING = "upload"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"



load_dotenv("/Users/marcalph/.ssh/llm_api_keys.env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)

def stress_dataprep(raw_path:Path) -> pd.DataFrame:
  run = wandb.init(project=WANDB_PROJECT, job_type=WandbJobType.UPLOAD.value)
  data = pd.read_csv(raw_path/"Reddit_Title.csv", sep=';')
  raw_artifact = wandb.Artifact(name="stress_raw", type=ArtifactType.RAW.value)
  raw_artifact.add_file(raw_path/"Reddit_Title.csv")

  proc_artifact = wandb.Artifact(name="stress_processed", type=ArtifactType.PREP.value)
  
  data_cleaned = data[['title', 'label']].head(5000)
  label_mapping = {0: "non-stress", 1: "stress"}
  data_cleaned['label'] = data_cleaned['label'].map(label_mapping)

  with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
    data_cleaned.to_csv(temp_csv.name, index=False)
    proc_artifact.add_file(temp_csv.name)

  run.log_artifact(raw_artifact)
  run.log_artifact(proc_artifact)
  run.finish()
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


def stress_split(df:pd.DataFrame, output_dir:Path, serialize:bool=True, fragment_trainset:bool=True) -> None:
  # split data into training and validation
  train_data, validation_data = train_test_split(df, test_size=0.2, random_state=42)
  
  if not serialize:
    return train_data, validation_data

  # serialize data
  os.makedirs(output_dir, exist_ok=True)
  run = wandb.init(project=WANDB_PROJECT, job_type=WandbJobType.SPLIT.value)
  ats  = {}
  if fragment_trainset:
    for i in np.linspace(len(train_data)//10, len(train_data), num=10):
      split = f'stress_detection_train_{int(i)}.jsonl'
      train_output_file_path = output_dir/split
      df_to_jsonl(train_data.iloc[:int(i)], train_output_file_path)
      ats[split] = wandb.Artifact(name=split, type=ArtifactType.TRAIN.value)
      ats[split].add_file(train_output_file_path)
  else:
    split = 'stress_detection_train.jsonl'
    train_output_file_path = output_dir/split
    df_to_jsonl(train_data, train_output_file_path)
    ats[split] = wandb.Artifact(name=split, type=ArtifactType.TRAIN.value)
    ats[split].add_file(train_output_file_path)

  split = 'stress_detection_validation.jsonl'
  validation_output_file_path = output_dir/split
  df_to_jsonl(validation_data, validation_output_file_path)
  ats[split] = wandb.Artifact(name=split, type=ArtifactType.VALIDATION.value)
  logger.info(f"jsonl files saved to {output_dir}")

  for _, at in ats.items():
    run.log_artifact(at)
  run.finish()
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
    training_file="file-ELcL7UfWJ11GKVcLMcX7rK", # train 800
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
    tags=["ft", "openai", "stress", "800"])
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


def eval_data_format(row):
    role_system_content = row["role: system"]
    role_system_dict = {"role": "system", "content": role_system_content}

    role_user_content = row["role: user"]
    role_user_dict = {"role": "user", "content": role_user_content}
    
    return [role_system_dict, role_user_dict]


if __name__ == "__main__":
  output_dir = STRESS_RAW_PATH.parent/"processed"
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MLEXPERIMENTS"))
  prepare_data = True
  train = False
  evaluate = False
  wandb_eval = False

  if prepare_data:
    # load data
    data_cleaned = stress_dataprep(STRESS_RAW_PATH)
    # process data
    stress_split(data_cleaned, output_dir, serialize=True, fragment_trainset=True)
    # upload
    # upload_datasets(output_dir, client)
  if train:
    for _ in range(3):
      finetune_test(client)
  if evaluate:
    data_cleaned = load_stress_data(STRESS_RAW_PATH)
    _, test_df = split_and_serialize_to_jsonl(data_cleaned, output_dir, serialize=False)
    y_pred = predict(client, test_df, model="gpt-3.5-turbo")  
    run_evaluation(test_df['label'], y_pred)
    y_pred = predict(client, test_df, model="ft:gpt-3.5-turbo-0125:wingmate::AarlN1qI")  
    run_evaluation(test_df['label'], y_pred)
  if wandb_eval:
    run = wandb.init(
      project='ml-experiments',
      job_type='eval'
    )
    VALIDATION_FILE_ARTIFACT_URI = 'wingmate-ai/ml-experiments/valid-file-JD8qkemtmDCdgH9xs9i5ce:v0'

    artifact_valid = run.use_artifact(
        VALIDATION_FILE_ARTIFACT_URI,
        type='validation_files'
    )
    artifact_valid_path = artifact_valid.download()
    logger.info(f"Downloaded the validation data at: {artifact_valid_path}")

    validation_file = glob.glob(f"{artifact_valid_path}/*.table.json")[0]
    with open(validation_file, 'r') as file:
        data = json.load(file)

    validation_df = pd.DataFrame(columns=data["columns"], data=data["data"])

    logger.info(f"There are {len(validation_df)} validation examples")
    run.config.update({"num_validation_samples":len(validation_df)})
    validation_df.head()
    validation_df["messages"] = validation_df.apply(lambda row: eval_data_format(row), axis=1)
    validation_df.head()

    MODEL_ARTIFACT_URI = 'wingmate-ai/ml-experiments/model-metadata:v8' # REPLACE THIS WITH YOUR OWN ARTIFACT URI

    model_artifact = run.use_artifact(
    MODEL_ARTIFACT_URI,
    type='model'
    )
    model_metadata_path = model_artifact.download()
    logger.info(f"Downloaded the validation data at: {model_metadata_path}")

    model_metadata_file = glob.glob(f"{model_metadata_path}/*.json")[0]
    with open(model_metadata_file, 'r') as file:
        model_metadata = json.load(file)

    fine_tuned_model = model_metadata["fine_tuned_model"]
    prediction_table = wandb.Table(columns=['messages', 'completion', 'target'])

    eval_data = []

    for idx, row in tqdm(validation_df.iterrows()):
        messages = row.messages
        target = row["role: assistant"]

        res = client.chat.completions.create(model=fine_tuned_model, messages=messages, max_tokens=10)
        completion = res.choices[0].message.content

        eval_data.append([messages, completion, target])
        prediction_table.add_data(messages[1]['content'], completion, target)

    wandb.log({'predictions': prediction_table})
    correct = 0
    for e in eval_data:
      if e[1].lower() == e[2].lower():
        correct+=1

    accuracy = correct / len(eval_data)

    print(f"Accuracy is {accuracy}")
    wandb.log({"eval/accuracy": accuracy})
    wandb.summary["eval/accuracy"] = accuracy

    correct = 0
    for e in eval_data:
      if e[1].lower() == e[2].lower():
        correct+=1

    accuracy = correct / len(eval_data)

    print(f"Accuracy is {accuracy}")
    wandb.log({"eval/accuracy": accuracy})
    wandb.summary["eval/accuracy"] = accuracy


    baseline_prediction_table = wandb.Table(columns=['messages', 'completion', 'target'])

    baseline_eval_data = []

    for idx, row in tqdm(validation_df.iterrows()):
        messages = row.messages
        target = row["role: assistant"]

        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=10)
        completion = res.choices[0].message.content

        baseline_eval_data.append([messages, completion, target])
        baseline_prediction_table.add_data(messages[1]['content'], completion, target)

    wandb.log({'baseline_predictions': baseline_prediction_table})
    baseline_correct = 0
    for e in baseline_eval_data:
      if e[1].lower() == e[2].lower():
        baseline_correct+=1

    baseline_accuracy = baseline_correct / len(baseline_eval_data)
    print(f"Baseline Accurcy is: {baseline_accuracy}")
    wandb.log({"eval/baseline_accuracy": baseline_accuracy})
    wandb.summary["eval/baseline_accuracy"] =  baseline_accuracy
    wandb.finish()



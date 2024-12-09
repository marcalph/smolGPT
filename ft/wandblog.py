from wandb.integration.openai.fine_tuning import WandbLogger
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv("/Users/marcalph/.ssh/llm_api_keys.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MLEXPERIMENTS"))
print(os.getenv("OPENAI_API_KEY_MLEXPERIMENTS"))

for ftjobid in ["ftjob-aqxaYcDNC37q8ovlAsFxsMl9", "ftjob-NsXD9NTgQr7Ych4R3VHEYxIy", "ftjob-z0NqIaA0nJT4DbmJjWhgQYQc"]: 
  WandbLogger.sync(
    fine_tune_job_id=ftjobid, 
    openai_client=client,
    overwrite=True,
    project="ml-experiments", 
    tags=["ft", "openai", "stress", "800"])


# WandbLogger.sync( openai_client=client,
#     overwrite=True,
#     project="ml-experiments", 
#     tags=["ft", "openai", "stress", "400"])
import os
from datasets import load_dataset
from tqdm import tqdm
from orign import ChatModel 

BASE_MODEL_NAME = "allenai/Molmo-7B-D-0924"
TUNED_MODEL_NAME = "agentsea/molmo-7b-ft-tideui"
PROVIDER = "vllm"

if __name__ == "__main__":
    os.makedirs("../../tmp/evals/molmo_general_clicking/", exist_ok=True)
    # load data
    ds_eval = load_dataset("agentsea/anchor", split="test")
    # load and connect to base model
    model = ChatModel(model=BASE_MODEL_NAME, provider=PROVIDER)
    model.connect()
    # run predictions with base model
    predictions = []
    targets = []
    resolutions = []
    for example in tqdm(ds_eval):
        image = example["image"]
        name = example["name"]
        
        response = model.chat(msg="Point to " + name, image=image)
        print(response.choices[0].text)
        break

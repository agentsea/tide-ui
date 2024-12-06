import os
from datasets import load_dataset
from tqdm import tqdm
from orign import ChatModel 

MODEL_NAME = "allenai/Molmo-7B-D-0924"
PROVIDER = "vllm"

if __name__ == "__main__":
    os.makedirs("../../tmp/evals/molmo_general_clicking/", exist_ok=True)
    # load data
    ds_eval = load_dataset("agentsea/anchor", split="test")
    # load and connect to model
    model = ChatModel(model=MODEL_NAME, provider=PROVIDER)
    model.connect()
    # run predictions
    predictions = []
    targets = []
    resolutions = []
    for example in tqdm(ds_eval):
        image = example["image"]
        name = example["name"]
        response = model.chat(msg="Point to " + name, image=image)
        print(response)
        break

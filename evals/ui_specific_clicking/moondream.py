import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import calculate_normalized_distance

NUM_EXAMPLES = 1024
BASE_MODEL_ID = "vikhyatk/moondream-next"


if __name__ == "__main__":
    os.makedirs("../../tmp/evals/ui_specific_clicking/", exist_ok=True)
    # load data
    ds_eval = load_dataset("agentsea/anchor", split="test")
    # load model and processor
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    # run predictions
    predictions = []
    targets = []
    resolutions = []
    for example in tqdm(ds_eval):
        image = example["image"]
        name = example["name"]
        res = model.point(image, name, tokenizer)
        try:
            point = [res[0]["x"] * image.width, res[0]["y"] * image.height]
        except Exception as e:
            point = [0, 0]
        predictions.append(point)
        targets.append(example["coordinates"])
        resolutions.append(example["resolution"])
    # save results
    with open(
        f"../../tmp/evals/ui_specific_clicking/{BASE_MODEL_ID.replace('/', '_')}.json",
        "w",
    ) as f:
        json.dump({"predictions": predictions, "targets": targets}, f)
    # calculate distances
    distances = list(
        map(
            lambda x: calculate_normalized_distance(x[0], x[1], x[2]),
            zip(predictions, targets, resolutions),
        )
    )
    print(
        f"Average normalized distance for {BASE_MODEL_ID}: {sum(distances) / len(distances)}"
    )

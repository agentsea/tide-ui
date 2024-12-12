import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import calculate_normalized_distance

NUM_EXAMPLES = 1024
MODEL_ID = "vikhyatk/moondream-next"


if __name__ == "__main__":
    os.makedirs("../../tmp/evals/general_clicking/", exist_ok=True)
    # load data
    ds_eval = load_dataset("agentsea/tide-ui", split="test").shuffle(seed=42)
    ds_eval = ds_eval.select(range(NUM_EXAMPLES))
    # load model and processor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # run predictions
    predictions = []
    targets = []
    resolutions = []
    for example in tqdm(ds_eval):
        image = example["image"]
        name = example["name"]
        res = model.point(image, name, tokenizer)
        point = [res[0]["x"] * image.width, res[0]["y"] * image.height]
        predictions.append(point)
        targets.append(example["point"])
        resolutions.append(example["resolution"])
    # save results
    with open(
        f"../../tmp/evals/general_clicking/{MODEL_ID.replace('/', '_')}.json",
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
        f"Average normalized distance for {MODEL_ID}: {sum(distances) / len(distances)}"
    )

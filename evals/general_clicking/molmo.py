import json
import os
import re
from typing import List

from datasets import load_dataset
from orign import ChatModel
from pydantic import BaseModel
from tqdm import tqdm
from utils import calculate_normalized_distance

BASE_MODEL_NAME = "allenai/Molmo-7B-D-0924"
PROVIDER = "vllm"
NUM_EXAMPLES = 1024


class Point(BaseModel):
    x: float | int
    y: float | int


def extract_point_from_molmo_response(response: str, resolution: List[int]) -> Point:
    pattern = r'<point x="(\d+(?:\.\d+)?)" y="(\d+(?:\.\d+)?)"'
    match = re.search(pattern, response)
    if match:
        x = float(match.group(1)) / 100 * resolution[0]
        y = float(match.group(2)) / 100 * resolution[1]
        return Point(x=x, y=y)
    else:
        return Point(x=0, y=0)


if __name__ == "__main__":
    os.makedirs("../../tmp/evals/general_clicking/", exist_ok=True)
    # load data
    ds_eval = load_dataset("agentsea/tide-ui", split="test").shuffle(seed=42)
    ds_eval = ds_eval.select(range(NUM_EXAMPLES))
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
        point = extract_point_from_molmo_response(
            response.choices[0].text, example["resolution"]
        )
        predictions.append([point.x, point.y])
        targets.append(example["point"])
        resolutions.append(example["resolution"])
    # save results
    with open(f"../../tmp/evals/general_clicking/{BASE_MODEL_NAME}.json", "w") as f:
        json.dump({"predictions": predictions, "targets": targets}, f)
    # calculate distances
    distances = list(
        map(
            lambda x: calculate_normalized_distance(x[0], x[1], x[2]),
            zip(predictions, targets, resolutions),
        )
    )
    print(
        f"Average normalized distance for {BASE_MODEL_NAME}: {sum(distances) / len(distances)}"
    )

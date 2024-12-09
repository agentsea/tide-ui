import json
import os
import re
from typing import List

from datasets import load_dataset
from orign import ChatModel
from pydantic import BaseModel
from tqdm import tqdm
from utils import calculate_normalized_distance

BASE_MODEL_NAME = "claude-3-5-sonnet-20241022"
PROVIDER = "litellm"
NUM_EXAMPLES = 1024
# for a fair comparison we should use this resolution with claude. for more info see: https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo#screen-size
TARGET_WIDTH = 1024
TARGET_HEIGHT = 768
PROMPT_TEMPLATE = """Detect the {element} in the image. Return its position using the following JSON format:

{{
    "x": $X,
    "y": $Y

}}

Full schema:

{schema}

Do not use code blocks. Return the JSON string only."""


class Point(BaseModel):
    x: float | int
    y: float | int


if __name__ == "__main__":
    os.makedirs("../../tmp/evals/sonnet_general_clicking/", exist_ok=True)
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
    schema = Point.model_json_schema()
    for example in tqdm(ds_eval):
        image = example["image"]
        name = example["name"]
        prompt = PROMPT_TEMPLATE.format(element=name, schema=schema)
        print(prompt)
        import pdb; pdb.set_trace()
        response = model.chat(msg=prompt, image=image)
        print(response)
        # predictions.append([point.x, point.y])
        # targets.append(example["point"])
        # resolutions.append(example["resolution"])
        break
    exit()

    # save results
    with open("../../tmp/evals/sonnet_general_clicking/predictions.json", "w") as f:
        json.dump({"predictions": predictions, "targets": targets}, f)

    # calculate distances
    distances = list(
        map(
            lambda x: calculate_normalized_distance(x[0], x[1], x[2]),
            zip(predictions, targets, resolutions),
        )
    )
    print(f"Mean std distance: {sum(distances) / len(distances)}")

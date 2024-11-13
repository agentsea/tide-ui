import argparse
import json
import os
import re

from datasets import load_dataset
from mllm import Router, RouterConfig
from pydantic import BaseModel
from threadmem import RoleThread
from tqdm import tqdm
from utils import image_to_b64, calculate_normalized_distance
from typing import List


class Point(BaseModel):
    x: float | int
    y: float | int


PROMPT_TEMPLATE = """Detect the {element} in the image. Return its position using the following JSON format:

{{
    "x": $X,
    "y": $Y

}}

Full schema:

{schema}

Do not use code blocks. Return the JSON string only."""

MOLMO_PROMPT_TEMPLATE = """Point to {element}"""


def extract_point_from_molmo_response(response: str, resolution: List[int]) -> Point:
    pattern = r'<point x="(\d+(?:\.\d+)?)" y="(\d+(?:\.\d+)?)"'
    match = re.search(pattern, response)
    if match:
        x = float(match.group(1)) / 100 * resolution[0]
        y = float(match.group(2)) / 100 * resolution[1]
        return Point(x=x, y=y)
    else:
        return Point(x=0, y=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a closed model on the WaveUI test split"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to evaluate",
        required=True,
        choices=[
            "anthropic/claude-3-5-sonnet-20241022",
            "hosted_vllm/allenai/Molmo-7B-D-0924",
        ],
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1024,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../tmp/evals/",
        help="Output file for predictions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    if args.model == "gemini/gemini-1.5-pro-latest":
        output_file_name = "evals_closed_gemini.json"
    elif args.model == "anthropic/claude-3-5-sonnet-20241022":
        output_file_name = "evals_closed_claude.json"
    elif args.model == "gpt-4o":
        output_file_name = "evals_closed_gpt.json"
    elif args.model == "hosted_vllm/allenai/Molmo-7B-D-0924":
        output_file_name = "evals_closed_molmo.json"
    ds = load_dataset("agentsea/wave-ui-points", split="test").shuffle(seed=42)
    examples = ds.select(range(args.num_examples))

    if args.model == "hosted_vllm/allenai/Molmo-7B-D-0924":
        custom_model = RouterConfig(
            model=args.model,
            api_base="https://models.agentlabs.xyz/v1",
            api_key_name="MOLMO_API_KEY",
        )
        router = Router(custom_model)
    else:
        router = Router(
            preference=[args.model],
        )

    schema = Point.model_json_schema()

    predictions = []
    targets = []
    resolutions = []

    for example in tqdm(examples):
        image = example["image"]
        original_image = image.copy()
        if args.model == "anthropic/claude-3-5-sonnet-20241022":
            target_width = 1024
            target_height = 768
            image = image.resize((target_width, target_height))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        img_b64 = image_to_b64(image, image_format="JPEG")

        prompt = (
            MOLMO_PROMPT_TEMPLATE.format(element=example["name"])
            if args.model == "hosted_vllm/allenai/Molmo-7B-D-0924"
            else PROMPT_TEMPLATE.format(element=example["name"], schema=schema)
        )

        thread = RoleThread()
        thread.post(
            role="user",
            msg=prompt,
            images=[img_b64],
        )

        try:
            if args.model == "hosted_vllm/allenai/Molmo-7B-D-0924":
                expect = None
            else:
                expect = Point
            response = router.chat(thread, expect=expect, retries=0)
            if args.model == "hosted_vllm/allenai/Molmo-7B-D-0924":
                point = extract_point_from_molmo_response(
                    response.msg.text, example["resolution"]
                )
            else:
                point = response.parsed
            # rescale if claude
            if args.model == "anthropic/claude-3-5-sonnet-20241022":
                original_width, original_height = original_image.size
                x = point.x * original_width / target_width
                y = point.y * original_height / target_height
                point = Point(x=x, y=y)
            x = point.x
            y = point.y

            predictions.append([x, y])
            targets.append(example["point"])
            resolutions.append(example["resolution"])
        except Exception as e:
            print(f"Skipping example due to error: {e}")
    with open(args.output_dir + output_file_name, "w") as f:
        json.dump({"predictions": predictions, "targets": targets}, f)

    distances = list(
        map(
            lambda x: calculate_normalized_distance(x[0], x[1], x[2]),
            zip(predictions, targets, resolutions),
        )
    )

    print(f"Mean std distance: {sum(distances) / len(distances)}")

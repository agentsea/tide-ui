import argparse
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)
from utils import calculate_normalized_distance
from typing import List
import re

def extract_coords(input_str: str) -> List[int]:
    pattern = r"<loc(\d{4})>"
    try:
        y1, x1, y2, x2 = [int(coord) for coord in re.findall(pattern, input_str)]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        out = [center_x, center_y]
    except Exception:
        out = [0, 0]
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an open model on the WaveUI test split"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to evaluate",
        required=True,
        choices=[
            "agentsea/paligemma-3b-ft-widgetcap-waveui-448",
            "agentsea/paligemma-3b-ft-waveui-896",
        ],
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1024,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../tmp/evals/",
        help="Output file for predictions",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="Precision to use for inference",
        default="fp32",
        choices=["fp32", "fp16", "int8", "nf4"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    model_id = args.model
    precision = args.precision

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    if model_id == "agentsea/paligemma-3b-ft-widgetcap-waveui-448":
        output_file_name = f"evals_open_paligemma_widgetcap_448_{precision}.json"
        processor_id = "google/paligemma-3b-pt-448"
    elif model_id == "agentsea/paligemma-3b-ft-waveui-896":
        output_file_name = f"evals_open_paligemma_896_{precision}.json"
        processor_id = "google/paligemma-3b-pt-896"

    # load model and processor
    if precision == "fp32":
        model = (
            PaliGemmaForConditionalGeneration.from_pretrained(model_id)
            .eval()
            .to("cuda")
        )
    elif precision == "fp16":
        model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )
            .eval()
            .to("cuda")
        )
    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_config, device_map={"": 0}
        ).eval()
    elif precision == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_config, device_map={"": 0}
        ).eval()
    else:
        raise ValueError("Invalid precision")

    processor = AutoProcessor.from_pretrained(processor_id)

    # load data
    ds = load_dataset("agentsea/wave-ui-paligemma", split="test")
    examples = ds.select(range(args.num_examples))

    predictions = []
    targets = []
    resolutions = []

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i : i + batch_size]

        tokens = processor(
            text=["detect " + name for name in batch["name"]],
            images=batch["image"],
            return_tensors="pt",
            padding="longest",
            tokenize_newline_separately=False,
        ).to(model.device)

        input_len = tokens["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**tokens, max_new_tokens=100, do_sample=False)
            generation = generation[:, input_len:]
            decoded = processor.batch_decode(generation, skip_special_tokens=True)

        if len(decoded) == len(batch["point"]):
            predictions.extend(decoded)
            targets.extend(batch["point"])
            resolutions.extend(batch["resolution"])

    predictions = list(map(extract_coords, predictions))
    target_error_indices = []
    prediction_error_indices = []
    for val in zip(predictions, targets):
        if val[0] == [0, 0]:
            prediction_error_indices.append(i)
        if val[1] == [0, 0]:
            target_error_indices.append(i)
    # filter out error indices
    predictions = [
        val
        for i, val in enumerate(predictions)
        if i not in target_error_indices and i not in prediction_error_indices
    ]
    targets = [
        val
        for i, val in enumerate(targets)
        if i not in target_error_indices and i not in prediction_error_indices
    ]
    print(f"Number of target errors: {len(target_error_indices)}")
    print(f"Number of prediction errors: {len(prediction_error_indices)}")
    with open(args.output_dir + output_file_name, "w") as f:
        json.dump({"predictions": predictions, "targets": targets}, f)

    distances = list(map(lambda x: calculate_normalized_distance(x[0], x[1], x[2]), zip(predictions, targets, resolutions)))

    print(f"Mean std distance: {sum(distances) / len(distances)}")

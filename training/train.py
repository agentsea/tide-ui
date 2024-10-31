from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

def point_to_xml(point: List[float], description: str = "") -> str:
    """Converts a point coordinate and description into XML format.

    Args:
        point (List[float]): A list containing x,y coordinates as floats.
        description (str, optional): Description text for the point. Defaults to "".

    Returns:
        str: XML string representing the point with coordinates and description.
    """
    x, y = point
    return f' <point x="{x:.1f}" y="{y:.1f}" alt="{description}">{description}</point>'

def normalize_point(point: List[float], resolution: List[int]) -> List[float]:
    """Normalizes point coordinates by dividing by the corresponding resolution values.

    Args:
        point (List[float]): A list containing x,y coordinates to normalize.
        resolution (List[int]): A list containing width,height values to normalize against.

    Returns:
        List[float]: Normalized coordinates as a list of floats.
    """
    return [p / r for p, r in zip(point, resolution)]


def data_collator(dataset, processor):
    """Collates dataset examples into batched model inputs.

    Args:
        dataset: Dataset containing examples with 'image', 'name', and 'point' fields.
        processor: Processor for converting text and images into model inputs.

    Returns:
        dict: Batched inputs with stacked tensors for model training, containing keys
            from the processor outputs with corresponding stacked tensor values.
    """
    inputs_list = []
    # TODO: we should be able to process in batch!
    for example in dataset:
        image = example["image"]
        question = "point to " + example["name"]
        answer = point_to_xml(normalize_point(example["point"], image.size), example["name"])
        text = "User: " + question + " Assistant:" + answer
        example_inputs = processor.process(text=text, images=[image], return_tensors="pt", padding=True, message_format=None)
        inputs_list.append(example_inputs)
    inputs = {
        k: torch.stack([inp[k] for inp in inputs_list])
        for k in inputs_list[0].keys()
    }
    return inputs


def train() -> None:
    """Trains the Molmo-7B-D model on the tide-ui dataset.

    This function loads the Molmo-7B-D model and processor, sets up training arguments,
    and trains the model on the tide-ui dataset using the Hugging Face Trainer.

    Args:
        None

    Returns:
        None: The trained model is saved to the specified output directory.
    """
    dataset = load_dataset("agentsea/tide-ui")
    # TODO: add more training args
    training_args = TrainingArguments(
        output_dir="../tmp/molmo-7b-d-0924",  # store in tmp
    )
    model_name = "allenai/Molmo-7B-D-0924"
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator(processor=processor),
    )

    trainer.train()


if __name__ == "__main__":
    train()

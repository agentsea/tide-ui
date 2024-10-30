from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

def point_to_xml(point: List[float], description: str = "") -> str:
    x, y = point
    return f' <point x="{x:.1f}" y="{y:.1f}" alt="{description}">{description}</point>'


def data_collator(dataset, processor):
    # TODO: finish this
    for example in dataset:
        image = example["image"]
        question = "point to " + example["name"]
        answer = point_to_xml(example["point"], example["name"])
        text = "User: " + question + " Assistant:" + answer
        inputs = processor.process(text=text, images=[image], return_tensors="pt", padding=True, message_format=None)
    # TODO: need to stack the inputs here, somehow
    return inputs


def train():
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

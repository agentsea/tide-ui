from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

from .utils import normalize_point, point_to_xml


def process_batch(
    processor: AutoProcessor,
    prompts: List[str],
    answers: List[str],
    images_list: List[List[Image.Image]],
) -> Dict[str, torch.Tensor]:
    """Process a batch of prompts, answers and images for model input.

    Args:
        processor: The model processor for tokenization and image processing
        prompts: List of text prompts to process
        answers: List of text answers to process
        images_list: List of lists of PIL images to process

    Returns:
        Dict with padded input_ids, images, image_input_idx, image_masks.
    """
    batch_size = len(prompts)
    assert batch_size == len(
        answers
    ), "Number of prompts, answers, and image lists must match"
    assert batch_size == len(
        images_list
    ), "Number of prompts, answers, and image lists must match"
    # define image processing kwargs
    images_kwargs = {
        "max_crops": 12,
        "overlap_margins": [4, 4],
        "base_image_input_size": [336, 336],
        "image_token_length_w": 12,
        "image_token_length_h": 12,
        "image_patch_size": 14,
        "image_padding_mask": True,
    }

    # process texts
    tokens_list = []
    for prompt, answer in zip(prompts, answers):
        tokens = processor.tokenizer.encode(
            " "  # assuming always_start_with_space=True
            + "User: "
            + prompt
            + " Assistant:"
            + answer,
            add_special_tokens=False,
        )
        tokens_list.append(tokens)

    # process images
    images_arrays_list = []
    image_idxs_list = []
    for images in images_list:
        if images:
            image_arrays = []
            for image in images:
                assert isinstance(image, Image.Image), "All images must be PIL images"
                image = image.convert("RGB")
                image = ImageOps.exif_transpose(image)
                image_arrays.append(np.array(image))
            images_arrays_list.append(image_arrays)
            image_idx = [-1] * len(image_arrays)
            image_idxs_list.append(image_idx)
        else:
            images_arrays_list.append(None)
            image_idxs_list.append(None)

    # multimodal preprocess each example
    outputs_list = []
    for i in range(batch_size):
        tokens = tokens_list[i]
        images = images_arrays_list[i]
        image_idx = image_idxs_list[i]
        out = processor.image_processor.multimodal_preprocess(
            images=images,
            image_idx=image_idx,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=1536,
            image_patch_token_id=processor.special_token_ids["<im_patch>"],
            image_col_token_id=processor.special_token_ids["<im_col>"],
            image_start_token_id=processor.special_token_ids["<im_start>"],
            image_end_token_id=processor.special_token_ids["<im_end>"],
            **images_kwargs,
        )
        outputs_list.append(out)

    # collate outputs into batched tensors, with added padding
    batch_outputs = {}
    for key in outputs_list[0].keys():
        tensors = [torch.from_numpy(out[key]) for out in outputs_list]
        batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=-1
        )

    # prepend BOS token
    batch_outputs["input_ids"] = torch.nn.functional.pad(
        batch_outputs["input_ids"],
        (1, 0),
        value=processor.tokenizer.eos_token_id,  # use eos token as bos token
    )
    # shift image input indices because of BOS token
    image_input_idx = batch_outputs["image_input_idx"]
    batch_outputs["image_input_idx"] = torch.where(
        image_input_idx < 0, image_input_idx, image_input_idx + 1
    )

    return batch_outputs


def data_collator(dataset, processor):
    """Collates dataset examples into batched model inputs.

    Args:
        dataset: Dataset containing examples with 'image', 'name', and 'point' fields.
        processor: Processor for converting text and images into model inputs.

    Returns:
        dict: Batched inputs with stacked tensors for model training, containing keys
            from the processor outputs with corresponding stacked tensor values.
    """
    # get texts and images
    prompts = ["Point to the " + row["name"] for row in dataset]
    answers = [
        point_to_xml(normalize_point(row["point"], row["image"].size), row["name"])
        for row in dataset
    ]
    images_list = [[row["image"]] for row in dataset]
    batch_outputs = process_batch(processor, prompts, answers, images_list)
    return batch_outputs


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

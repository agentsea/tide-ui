import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import List, Dict

processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto",
)

# load 4 random examples from the dataset
data = load_dataset("agentsea/tide-ui", split="train", num_proc=8)
data = data.shuffle(seed=42).select(range(4))

def process_batch(processor: AutoProcessor, 
                 texts: List[str], 
                 images_list: List[List[Image.Image]]) -> Dict[str, torch.Tensor]:
    """Process a batch of texts and images for model input.

    Args:
        processor: The model processor for tokenization and image processing
        texts: List of text prompts to process
        images_list: List of lists of PIL images to process

    Returns:
        Dict with padded input_ids, images, image_input_idx, image_masks.
    """
    batch_size = len(texts)
    assert batch_size == len(images_list), "Number of texts and images must match"
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
    for text in texts:
        tokens = processor.tokenizer.encode(
            " " # assuming always_start_with_space=True
            + "User: "
            + text
            + " Assistant:",
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
        batch_outputs["input_ids"], (1, 0), value=processor.tokenizer.eos_token_id # use eos token as bos token
    )
    # shift image input indices because of BOS token
    image_input_idx = batch_outputs["image_input_idx"]
    batch_outputs["image_input_idx"] = torch.where(
        image_input_idx < 0, image_input_idx, image_input_idx + 1
    )

    return batch_outputs


# `process_batch` expects a list of texts and a **list of lists** of images
texts = ["Point to the " + example["name"] for example in data]
images_list = [[example["image"]] for example in data]

inputs = process_batch(processor, texts, images_list)

# move inputs to device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# batch generate
output = model.generate_from_batch(
    inputs,
    GenerationConfig(
        max_new_tokens=200,
        stop_sequences=["<|endoftext|>"],
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    ),
    tokenizer=processor.tokenizer,
)

# batch decode
generated_texts = processor.tokenizer.batch_decode(
    output[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
)

for prompt, text in zip(texts, generated_texts):
    print(f"\nPrompt: {prompt}")
    print(f"Response: {text}")

for i, example in enumerate(data):
    print(
        f"Example {i+1}: {[p / r for p, r in zip(example['point'], example['resolution'])]}"
    )

import torch
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_id = "agentsea/paligemma-3b-ft-waveui-896"
lora_rank = 8
num_epochs = 4
batch_size = 1
learning_rate = 0.0001
gradient_accumulation_steps = 8
warmup_steps = 2
weight_decay = 0.000001
adam_beta2 = 0.999
logging_steps = 1
save_steps = 2000
save_total_limit = 1
dataloader_pin_memory = False
push_to_hub = False


def point2pg_output(point, res):
    """Convert to PaliGemma format: <locdddd><locddd><locddd><locddd> string"""
    x1, y1, x2, y2 = point
    width, height = res
    x1 = "<loc" + str(int(x1 / width * 1024)).zfill(4) + ">"
    y1 = "<loc" + str(int(y1 / height * 1024)).zfill(4) + ">"
    x2 = "<loc" + str(int(x2 / width * 1024)).zfill(4) + ">"
    y2 = "<loc" + str(int(y2 / height * 1024)).zfill(4) + ">"
    return y1 + x1 + y2 + x2


def collate_fn(examples):
    image_token = "<image>"
    bos_token = processor.tokenizer.bos_token

    # Add image tokens and BOS token explicitly to the input text
    texts = [
        f"{image_token}{bos_token}detect {example['element_name']}"
        for example in examples
    ]
    # texts = ["detect " + example["element_name"] for example in examples]
    labels = [
        point2pg_output(example["bbox"], example["resolution"])
        + f" {example['element_name']}"
        for example in examples
    ]
    images = [example["image"] for example in examples]
    # print(texts)
    # print(labels)
    # print(images)
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
        # tokenize_newline_separately=False,
    )

    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens


ds = load_dataset("agentsea/anchor")
train_ds = ds["train"]

processor = PaliGemmaProcessor.from_pretrained(base_model_id)

# # Test the collate function with the first 3 examples
# test_examples = [train_ds[i] for i in range(3)]

# # Apply collate_fn to the test examples
# test_batch = collate_fn(test_examples)


# # Print some information about the generated batch to verify the output
# print(test_batch.keys()) # Print the keys of the returned dictionary
# print(test_batch["input_ids"].shape) # Print the shape of input_ids tensor
# # Print the first few input ids
# print(test_batch["input_ids"][0,:10])

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.bfloat16,
)

lora_config = LoraConfig(
    r=lora_rank,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map={"": 0}
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_epochs,
    remove_unused_columns=False,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    adam_beta2=adam_beta2,
    logging_steps=logging_steps,
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=save_steps,
    push_to_hub=push_to_hub,
    save_total_limit=save_total_limit,
    bf16=True,
    dataloader_pin_memory=dataloader_pin_memory,
    report_to=None,
    hub_private_repo=True,
)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    data_collator=collate_fn,
    args=training_args,
)

trainer.train()

trainer.push_to_hub("agentsea/pg-airbnb-test")

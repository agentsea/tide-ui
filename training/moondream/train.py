from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from datasets import load_dataset
import torch
import os
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from bitsandbytes.optim import Adam8bit
from torch.utils.data import Dataset


os.environ["WANDB_PROJECT"] = "moondream-next-tideui"
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 1
LR = 1e-5
USE_WANDB = False
DEVICE = "cuda"
ANSWER_EOS = "<|endoftext|>"
IMG_TOKENS = 729

# load model and tokenizer
model_name = "vikhyatk/moondream-next"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map={"": DEVICE},
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class AnchorDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset("agentsea/anchor", trust_remote_code=True)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        normalized_coords = [sample['coordinates'][0] / sample["image"].width, sample['coordinates'][1] / sample["image"].height]
        return {
            "image": sample["image"],  # PIL image
            "qa": [
                {
                    "name": sample['name'],
                    "point": f"{normalized_coords}",  # TODO: update this to use the correct format
                }
            ],
        }


datasets = {
    "train": AnchorDataset("train"),
    "test": AnchorDataset("test"),
}

sample = datasets['train'][0]

for qa in sample['qa']:
    print('Question:', qa['name'])
    print('Ground Truth:', qa['point'])
    print('Moondream:', model.point(
        sample['image'],
        qa['name'],
        tokenizer=tokenizer,
    ))
import pdb; pdb.set_trace()


def collate_fn(batch):
    images = [sample["image"] for sample in batch]
    images = [model.vision_encoder.preprocess(image) for image in images]

    labels_acc = []
    tokens_acc = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        for qa in sample["qa"]:
            q_t = tokenizer(
                f"\n\nPoint: {qa['name']}\n\nAnswer:", add_special_tokens=False
            ).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))

            a_t = tokenizer(
                f" {qa['point']}{ANSWER_EOS}", add_special_tokens=False
            ).input_ids
            toks.extend(a_t)
            labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)

    max_len = -1
    for labels in labels_acc:
        max_len = max(max_len, len(labels))

    attn_mask_acc = []

    for i in range(len(batch)):
        len_i = len(labels_acc[i])
        pad_i = max_len - len_i

        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    return (
        images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )


def compute_loss(batch, model):
    images, tokens, labels, attn_mask = batch

    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = model.vision_encoder(images)

    tok_embs = model.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat(
        (tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1
    )

    outputs = model.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


dataloaders = {
    "train": DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
}
model.text_model.train()
model.text_model.transformer.gradient_checkpointing_enable()
# run training
total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
optimizer = Adam8bit(
    [
        {"params": model.text_model.parameters()},
    ],
    lr=LR * 0.1,
    betas=(0.9, 0.95),
    eps=1e-6,
)

if USE_WANDB:
    import wandb

    wandb.init(
        project="moondream-ft",
        config={
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
    )

i = 0
for epoch in range(EPOCHS):
    for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        i += 1

        loss = compute_loss(batch, model)
        loss.backward()

        if i % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if USE_WANDB:
            wandb.log(
                {"loss/train": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            )

if USE_WANDB:
    wandb.finish()

model.save_pretrained("checkpoints/moondream-next-ft-tideui")
tokenizer.push_to_hub("agentsea/moondream-next-ft-tideui", private=True)
# run eval loop
model.eval()

# for i, sample in enumerate(datasets['test']):
#     md_answer = model.answer_question(
#         model.encode_image(sample['image']),
#         sample['qa'][0]['question'],
#         tokenizer=tokenizer,
#         num_beams=4,
#         no_repeat_ngram_size=5,
#         early_stopping=True
#     )

#     if i < 3:
#         print('Question:', sample['qa'][0]['question'])
#         print('Ground Truth:', sample['qa'][0]['answer'])
#         print('Moondream:', md_answer)
#     else:
#         break

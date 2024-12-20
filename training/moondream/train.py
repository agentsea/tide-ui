import math
import os
from typing import List, Tuple

import datasets
import PIL
import torch
import torchvision
import transformers
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 1
LR = 1e-6
WEIGHT_DECAY = 0.001
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 2.0
USE_WANDB = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_TOKENS = 729
MODEL_ID = "vikhyatk/moondream-next"
PROJECT_NAME = "moondream-next-tideui"
EVAL_STEPS = 100
EVAL_BATCH_SIZE = 16


class PointDataset(Dataset):
    def __init__(self, split: str = "train"):
        self.data: datasets.Dataset = load_dataset(
            "agentsea/anchor", trust_remote_code=True
        )[split]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample: dict = self.data[idx]
        normalized_coords: List[float] = [
            sample["coordinates"][0] / sample["image"].width,
            sample["coordinates"][1] / sample["image"].height,
        ]
        return {
            "image": sample["image"],
            "points": normalized_coords,
            "query": sample["name"],
        }


def collate_fn(
    batch: List[dict], model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for the point dataset."""
    images: List[PIL.PngImagePlugin.PngImageFile] = [
        sample["image"] for sample in batch
    ]
    encoded_images: List[torchvision.tv_tensors._image.Image] = [
        model.vision_encoder.preprocess(image) for image in images
    ]
    labels_acc: List[List[int]] = []
    tokens_acc: List[List[int]] = []
    for sample in batch:
        toks: List[int] = [tokenizer.bos_token_id]
        labs: List[int] = [-100] * (IMG_TOKENS + 1)
        query_tokens: List[int] = tokenizer(
            f"\n\nPoint: {sample['query']}\n\n", add_special_tokens=False
        ).input_ids
        toks.extend(query_tokens)
        labs.extend([-100] * len(query_tokens))
        x_coord_token: int = 50257
        y_coord_token: int = 50257
        toks.append(x_coord_token)
        labs.append(int(sample["points"][0] * 1024))
        toks.append(y_coord_token)
        labs.append(int(sample["points"][1] * 1024))
        tokens_acc.append(toks)
        labels_acc.append(labs)
    max_len: int = max(len(l) for l in labels_acc)
    attn_mask_acc: List[List[int]] = []
    for i in range(len(batch)):
        len_i: int = len(labels_acc[i])
        pad_i: int = max_len - len_i
        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)
    return (
        encoded_images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )


def compute_loss(
    batch: Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
    model: AutoModelForCausalLM,
) -> torch.Tensor:
    """Compute the loss for the batch."""
    images, tokens, labels, attn_mask = batch
    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)
    with torch.no_grad():
        img_embs: torch.Tensor = model.vision_encoder(images)
    tok_embs: torch.Tensor = model.text_model.get_input_embeddings()(tokens)
    inputs_embeds: torch.Tensor = torch.cat(
        (tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1
    )
    outputs: transformers.modeling_outputs.CausalLMOutputWithPast = model.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )
    return outputs.loss


def lr_schedule(step: int, max_steps: int, base_lr: float) -> float:
    """Learning rate schedule."""
    x: float = step / max_steps
    if x < 0.1:
        return 0.1 * base_lr + 0.9 * base_lr * x / 0.1
    else:
        return 0.1 * base_lr + 0.9 * base_lr * (1 + math.cos(math.pi * (x - 0.1))) / 2

def evaluate(model: AutoModelForCausalLM, eval_dataset: Dataset, tokenizer: AutoTokenizer, batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Run evaluation on the test dataset with specified batch size."""
    model.eval()
    total_distance = 0
    total_samples = 0
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, model, tokenizer),
    )
    
    with torch.no_grad():
        for batch in eval_dataloader:
            loss = compute_loss(batch, model)
            total_distance += loss.item() * batch_size
            total_samples += batch_size
    
    model.train()
    return {
        "avg_loss": total_distance / total_samples if total_samples > 0 else float('inf'),
        "samples_evaluated": total_samples
    }


def main():
    if USE_WANDB:
        import wandb

        os.environ["WANDB_PROJECT"] = PROJECT_NAME
        wandb.init(
            project=PROJECT_NAME,
            config={
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
                "LR": LR,
            },
        )
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map={"": DEVICE},
        attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Creating datasets...")
    datasets = {
        "train": PointDataset("train"),
        "test": PointDataset("test"),
    }
    print("Creating dataloader...")
    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model, tokenizer),
    )
    model.text_model.train()
    model.text_model.transformer.gradient_checkpointing_enable()
    print("Initializing optimizer...")
    optimizer: torch.optim.Optimizer = AdamW(
        [{"params": model.text_model.parameters()}],
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=WEIGHT_DECAY,
    )
    num_training_steps = EPOCHS * len(train_dataloader) // GRAD_ACCUM_STEPS
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print("Starting training...")
    i = 0
    best_eval_loss = float('inf')
    for epoch in range(EPOCHS):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            i += 1
            loss = compute_loss(batch, model)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            if i % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if USE_WANDB:
                wandb.log({
                    "loss/train": loss.item() * GRAD_ACCUM_STEPS,
                    "lr": scheduler.get_last_lr()[0]
                })
            if i % EVAL_STEPS == 0:
                print(f"\nRunning evaluation at step {i}...")
                eval_metrics = evaluate(model, datasets["test"], tokenizer)
                print(f"Average eval loss: {eval_metrics['avg_loss']:.4f}")
                if USE_WANDB:
                    wandb.log({
                        "eval/avg_loss": eval_metrics['avg_loss'],
                        "eval/samples": eval_metrics['samples_evaluated']
                    })
                if eval_metrics['avg_loss'] < best_eval_loss:
                    best_eval_loss = eval_metrics['avg_loss']
                    print(f"New best model with average loss: {best_eval_loss:.4f}")
                    save_dir = f"../../tmp/{PROJECT_NAME}_best"
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
    print("Saving model...")
    save_dir: str = f"../../tmp/{PROJECT_NAME}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    if USE_WANDB:
        wandb.finish()
    print("Running evaluation...")
    model.eval()
    for i, sample in enumerate(datasets["test"]):
        if i >= 3:
            break
        points: List[dict] = model.point(
            sample["image"], sample["query"], tokenizer=tokenizer, max_objects=1
        )
        if points:
            predicted_point: dict = points[0]
            actual_point: List[float] = sample["points"]
            print(f"\nSample {i + 1}")
            print(f"Query: {sample['query']}")
            print(
                f"Predicted point: ({predicted_point['x']:.3f}, {predicted_point['y']:.3f})"
            )
            print(f"Actual point: ({actual_point[0]:.3f}, {actual_point[1]:.3f})")


if __name__ == "__main__":
    main()
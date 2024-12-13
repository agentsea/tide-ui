from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from typing import Dict, List
from datasets import load_dataset
import torch
import os

os.environ["WANDB_PROJECT"] = "moondream-next-tideui"


class DataCollator:
    """A data collator class for batching dataset examples into model inputs.

    This collator processes a list of dataset examples containing images, names and points
    into batched tensors suitable for model training. It formats prompts and answers,
    processes images, and uses the processor to convert everything into model inputs.

    Args:
        processor (AutoProcessor): The processor to use for converting text and images
            into model inputs.

    Attributes:
        processor (AutoProcessor): The stored processor instance.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collates dataset examples into batched model inputs.

        Args:
            dataset (List[Dict]): List of dataset examples, where each example is a dict
                containing 'image', 'name', and 'point' fields.

        Returns:
            Dict[str, torch.Tensor]: Batched inputs with stacked tensors for model training,
                containing keys from the processor outputs with corresponding stacked tensor
                values.
        """
        ANSWER_EOS = "<|endoftext|>" # TODO: maybe change this
        IMG_TOKENS = 729 # TODO: maybe change this
        # format images, prompts and answers
        images = [
            self.model.vision_encoder.preprocess(sample["image"]) for sample in batch 
        ]
        labels_acc = []
        tokens_acc = []
        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)
            q_t = tokenizer(
                f"\n\nQuestion: What do you see in the image?\n\nAnswer:",
                add_special_tokens=False
            ).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))
            a_t = tokenizer(
                f" A UI screenshot{ANSWER_EOS}",
                add_special_tokens=False
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
        print(f"tokens shape: {torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]).shape}")
        # return (
        #     images,
        #     torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        #     torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        #     torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        # )

        tokens = torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc])
        labels = torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc])
        attn_mask = torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc])
        # tokens = tokens.to("cuda")
        # labels = labels.to("cuda")
        # attn_mask = attn_mask.to("cuda")
        with torch.no_grad():
            img_embs = model.vision_encoder(images)

        tok_embs = model.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)
        return {
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "attention_mask": attn_mask,
        }



if __name__ == "__main__":
    # load data
    train_dataset = load_dataset("agentsea/anchor", split="train")
    eval_dataset = load_dataset("agentsea/anchor", split="test")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    # set training args
    training_args = TrainingArguments(
        output_dir="../../tmp/moondream-next-tideui",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # eval
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=10,
        # logging
        logging_steps=1,
        report_to="wandb",
        save_steps=1000,
        save_total_limit=1,
        hub_private_repo=True,
        hub_model_id="agentsea/moondream-next-ft-tideui",
        push_to_hub=False,  # TODO: maybe revert this in full run
        # opt. TODO: update parameters here
        learning_rate=1e-5,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.05,
        max_grad_norm=1.0,
        seed=3407,
    )
    # load model
    model_name = "vikhyatk/moondream-next"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # setup trainer
    trainer = Trainer(
        model=model.text_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(model, tokenizer),
    )
    # run training
    trainer.train()
    trainer.push_to_hub()
    tokenizer.push_to_hub("agentsea/moondream-next-ft-tideui", private=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from typing import Dict, List
from datasets import load_dataset
import torch


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

    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, dataset: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collates dataset examples into batched model inputs.

        Args:
            dataset (List[Dict]): List of dataset examples, where each example is a dict
                containing 'image', 'name', and 'point' fields.

        Returns:
            Dict[str, torch.Tensor]: Batched inputs with stacked tensors for model training,
                containing keys from the processor outputs with corresponding stacked tensor
                values.
        """
        # format images, prompts and answers
        prompts = []  # TODO: add prompts
        answers = []  # TODO: add answers
        images_list = [[row["image"]] for row in dataset]
        # batch process
        # TODO: add batch processing
        return  # TODO: add return


if __name__ == "__main__":
    # load data
    train_dataset = load_dataset("agentsea/anchor", split="train")
    eval_dataset = load_dataset("agentsea/anchor", split="test")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    # set training args
    training_args = TrainingArguments(
        output_dir="../../tmp/moondream-next-anchor",
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
        hub_model_id="agentsea/moondream-next-ft-anchor",
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
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    # run training
    trainer.train()
    trainer.push_to_hub()
    tokenizer.push_to_hub("agentsea/moondream-next-ft-anchor", private=True)

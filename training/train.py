import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments


def data_collator(dataset):
    # TODO: finish this
    texts = []
    images = []
    for example in dataset:
        image = example["image"]
        question = example["question"]
        answer = example["answer"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
      text = processor.apply_chat_template(messages, add_generation_prompt=False)
      texts.append(text.strip())
      images.append([image])
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    # TODO: some things missing here
    return batch


def train():
    dataset = load_dataset("agentsea/tide-ui")
    # TODO: add more training args
    training_args = TrainingArguments(
        output_dir="../tmp/molmo-7b-d-0924", # store in tmp
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
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    train()

# Training

```bash
torchrun --nproc-per-node $NUM_GPUS train.py
```

It seems to be working atm with at least 4 A100 80GB. Right now a good setup is having BS=2 with bf16. However, an alternative setup would be to have BS=3 with activation_checkpointing enabled and gradient_accumulation_steps=8.
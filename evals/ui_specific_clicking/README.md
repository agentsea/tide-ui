# UI-specific clicking

We run predictions over all examples in the test-set from the [anchor](https://huggingface.co/datasets/agentsea/anchor) dataset using different models and calculate the distance between the predictions and the ground-truth, normalizing by the image diagonal to account for different image resolutions.

Average normalized distance for allenai/Molmo-7B-D-0924: 0.07353734766234708

Average normalized distance for agentsea/molmo-7b-ft-tideui: 0.030793343330084516

The above shows an improvement of ~4% of the image diagonal between the finetuned model and the base model, indicating that the fine-tuning pipeline is promising as a potential path for improvement over agentic tasks in specific UIs.
# UI-specific clicking

We run predictions over all examples in the test-set from the [anchor](https://huggingface.co/datasets/agentsea/anchor) using different models and calculate the distance between the predictions and the ground-truth, normalizing by the image diagonal to account for different image resolutions.
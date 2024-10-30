# Data

The dataset used in this project is a subset of the original [WaveUI dataset](https://github.com/agentsea/wave-ui). The dataset is stored as a new [TideUI dataset](https://huggingface.co/datasets/agentsea/tide-ui).

In order to obtain the TideUI dataset we:

- Filtered-out all non-English examples from the WaveUI dataset.
- Removed duplicate boxes within the same image group.
- Converted the bounding boxes into points by taking the center-point of the box.
- Kept only the `image`, `resolution`, `name`, `point` columns.
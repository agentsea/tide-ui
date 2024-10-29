<p align="center">
  <img src="tideui.png" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/collections/agentsea/waveui-6684c5ab7b72cda3a523674c"> HF Collection</a>&nbsp
<br>

---

# TideUI

Data and tools to improve UI clicking.

This project is a follow-up on [WaveUI](https://github.com/agentsea/wave-ui).

## Data

The data used for this project is a subset of the [WaveUI dataset](https://huggingface.co/datasets/agentsea/wave-ui). It includes only the English examples of the dataset and the points were obtained by getting the center of the bounding boxes.

## Training

We currently focus on [Molmo](https://molmo.allenai.org/blog). This model has already been fine-tuned on the pointing task. In this project, we further fine-tune the model to improve its preformance on UI settings.

## Evaluation

We evaluate the models using the test-split of the TideUI dataset. Specifically, we measure how distant each model's predictions are from the ground truth and normalize the distance usintg each image's diagonal length to accound for variations in resolution.
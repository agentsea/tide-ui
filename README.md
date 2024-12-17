<p align="center">
  <img src="./assets/tideui.png" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/collections/agentsea/waveui-6684c5ab7b72cda3a523674c"> HF Collection</a>&nbsp
<br>

---

# TideUI

Data and tools to improve UI-specific clicking.

There are models today that perform well on the task of clicking on UI elements. Two of the main approaches that have been taken by language models are by detecting the elements with bounding boxes and with points. Both options have been proven to be paths towards high performance on generic UI clicking.

However, for many use-cases, it's not necessary for a model to be performant in **any** UI, but, rather, in a specific UI. For examples, someone looking for flights might be satisfied with a model that performs well on the Google Flights UI.

This project explores a pipeline to improve click models on specific UIs.

## Data

UI-specific click-data can be captured in a process similar to the one outlined in [data/](./data/). For more info, see the instructions there.

## Training

Among others, some of the best open-source models for clicking are:

1. [Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) 
2. [Moondream](https://github.com/vikhyat/moondream)

We will focus on these two models for now. For more info on their performance vs Claude Sonnet 3.5 see [evals/general_clicking/](./evals/general_clicking/)

## Evaluation

We run two evaluations:

1. UI-specific clicking - this is based on the test-set gathered from the specific UI. We calculate the mean normalized distance from ground truth and compare this to the base Molmo and Moondream models + Claude Sonnet 3.5, a closed source option with clicking capabilities.
2. UI-specific agent task - this quantifies how much better an agent performs in the specific UI over a range of tasks. 

For more details see [evals/](./evals/).

## TODO

- [ ] ds eval
- [ ] agent eval
- [ ] training experiments: change lr/epoch/ds_size/etc
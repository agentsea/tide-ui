<p align="center">
  <img src="tideui.png" width="400"/>
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

The gathering of the data has two components:

1. A process to capture screenshots of specific UI on various user stories.
2. A process to annotate the elements in theses screenshots.

For more information see [data/](./data/)

## Training

Among others, some of the best open-source models for clicking are:

1. [Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) 
2. [PaliGemma](https://huggingface.co/agentsea/paligemma-3b-ft-waveui-896)
3. [Moondream](https://github.com/vikhyat/moondream) (in the next release)
4. [ShowUI](https://github.com/showlab/ShowUI)

We will experiment with these models to see which one is best fit for this specific use-case. We care about performance and cost/efficiency.

## Evaluation

TBD


## TODO

- [ ] Get screenshots in full-screen.
- [ ] Use OmniParser's YOLO model for automatic bounding boxes.
- [ ] Use Sonnet 3.5 for automatic annotation
- [ ] Repeat fine-tune with PaliGemma
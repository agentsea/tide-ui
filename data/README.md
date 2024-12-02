# Data

There are two processes that are involved in the generation of the data for this projects:

1. A process to capture screenshots of the UI.
2. A process to annotate these screenshots.

For the former, one can run the `agentdesk.ipynb` notebook. This will launch a VM locally that will have a background loop that captures the screenshots.

Once we have the screenshots, we can annotate them with the gradio app in `bounding_box_annotator.py`.

The annotated images can then be processed into a HF dataset with `bb_preprocess.py`. An example dataset is currently hosted in [anchor](https://huggingface.co/datasets/agentsea/anchor) for the AirBnb interface.
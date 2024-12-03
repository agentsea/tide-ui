# Data

There are two processes that are involved in the generation of the data:

1. Collect screenshots from a specific UI
2. Annotate the elements in the screenshots

The first one can be done by running a process that gets screenshots while a user interacts with the UI. A simple PoC of how this can be done can be found in `screenshots.ipynb`

The second can be done by running the `annotator.py` Gradio app, which will collect the clicks a user makes and send those to Anthropic's Sonnet 3.5 to annotate. The annotations can be edited by the user.

Finally, the dataset can be pushed to HF so that it's stored for fine-tuning.
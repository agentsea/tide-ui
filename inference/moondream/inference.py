from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw
import requests
import torch

model_id = "vikhyatk/moondream-next"

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# inference
image = Image.open("airbnb_test_input.png")
points = model.point(image, "search button", tokenizer)
print(f"points: {points}")
# Draw point on image
draw = ImageDraw.Draw(image)
x = points[0]["x"] * image.width
y = points[0]["y"] * image.height
radius = 5
draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red")
image.save("airbnb_test_output.png")

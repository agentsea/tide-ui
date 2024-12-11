from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests

model_id = "vikhyatk/moondream-next"

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
# enc_image = model.encode_image(image) # TODO: maybe remove? 
print(model.point(image, "dog", tokenizer))
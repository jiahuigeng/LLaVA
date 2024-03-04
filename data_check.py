import pickle
import numpy as np
import requests
from PIL import Image
from io import BytesIO
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


image_tensor = pickle.load(open("prompt.bin", "rb"))
image_tensor_1 = load_image("prompt.png")
input_embeds = pickle.load(open("input_embeds.bin", "rb"))

out_padded_embeds = pickle.load(open("padded_prompt.bin", "rb"))

print(image_tensor)
print(image_tensor_1)
